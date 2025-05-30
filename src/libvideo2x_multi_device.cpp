#include "libvideo2x_multi_device.h"
#include <libavcodec/avcodec.h>
#include <queue>
#include <list>
#include <thread>

extern "C" {
    #include <libavutil/avutil.h>
    #include <libavfilter/avfilter.h>
    #include <libavfilter/buffersink.h>
    #include <libavfilter/buffersrc.h>
}

#include <spdlog/spdlog.h>

#include "avutils.h"
#include "decoder.h"
#include "encoder.h"
#include "logger_manager.h"
#include "processor.h"
#include "processor_factory.h"


namespace video2x {

struct FrameInfo {
    AVFrame* frame;
    int64_t frame_idx;
};

class UnsharpFilter {
public:
    AVFilterGraph* filter_graph = nullptr;
    AVFilterContext* buffersrc_ctx = nullptr;
    AVFilterContext* buffersink_ctx = nullptr;

    // Call this once, after you know width/height/pix_fmt
    int init(AVPixelFormat pix_fmt, int width, int height, AVRational time_base, const char* unsharp_params = "5:5:1.5") {
        char args[512];
        int ret = 0;

        filter_graph = avfilter_graph_alloc();
        if (!filter_graph) return AVERROR(ENOMEM);

        // Buffer source
        snprintf(args, sizeof(args),
                 "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=1/1",
                 width, height, pix_fmt, time_base.num, time_base.den);

        ret = avfilter_graph_create_filter(&buffersrc_ctx, avfilter_get_by_name("buffer"),
                                           "in", args, nullptr, filter_graph);
        if (ret < 0) return ret;

        // Buffer sink
        ret = avfilter_graph_create_filter(&buffersink_ctx, avfilter_get_by_name("buffersink"),
                                           "out", nullptr, nullptr, filter_graph);
        if (ret < 0) return ret;

        // Unsharp filter
        AVFilterContext* unsharp_ctx = nullptr;
        ret = avfilter_graph_create_filter(&unsharp_ctx, avfilter_get_by_name("unsharp"),
                                           "unsharp", unsharp_params, nullptr, filter_graph);
        if (ret < 0) return ret;

        // Link filters: buffer -> unsharp -> buffersink
        ret = avfilter_link(buffersrc_ctx, 0, unsharp_ctx, 0);
        if (ret < 0) return ret;
        ret = avfilter_link(unsharp_ctx, 0, buffersink_ctx, 0);
        if (ret < 0) return ret;

        ret = avfilter_graph_config(filter_graph, nullptr);
        return ret;
    }

    // For each frame
    int filter_frame(AVFrame* in, AVFrame* out) {
        int ret = av_buffersrc_add_frame_flags(buffersrc_ctx, in, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (ret < 0) return ret;
        return av_buffersink_get_frame(buffersink_ctx, out);
    }

    ~UnsharpFilter() {
        if (filter_graph) avfilter_graph_free(&filter_graph);
    }
};

VideoProcessorMultiDevice::VideoProcessorMultiDevice(
    const processors::ProcessorConfig proc_cfg,
    const encoder::EncoderConfig enc_cfg,
    const std::string &vk_device_list,
    const AVHWDeviceType hw_device_type,
    const bool benchmark,
    const std::string& filter_options
)
    : proc_cfg_(proc_cfg),
      enc_cfg_(enc_cfg),
      vk_device_list_(vk_device_list),
      hw_device_type_(hw_device_type),
      benchmark_(benchmark),
      filter_options_(filter_options) {}

[[gnu::target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")]]
int VideoProcessorMultiDevice::process(
    const std::filesystem::path in_fname,
    const std::filesystem::path out_fname
) {
    int ret = 0;

    // Helper lambda to handle errors:
    auto handle_error = [&](int error_code, const std::string& msg) {
        // Format and log the error message
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(error_code, errbuf, sizeof(errbuf));
        logger()->critical("{}: {}", msg, errbuf);

        // Set the video processor state to failed and return the error code
        state_.store(VideoProcessorMultiDeviceState::Failed);
        return error_code;
    };

    // Set the video processor state to running
    state_.store(VideoProcessorMultiDeviceState::Running);

    // Create a smart pointer to manage the hardware device context
    std::unique_ptr<AVBufferRef, decltype(&avutils::av_bufferref_deleter)> hw_ctx(
        nullptr, &avutils::av_bufferref_deleter
    );

    // Initialize hardware device context
    if (hw_device_type_ != AV_HWDEVICE_TYPE_NONE) {
        AVBufferRef* tmp_hw_ctx = nullptr;
        ret = av_hwdevice_ctx_create(&tmp_hw_ctx, hw_device_type_, nullptr, nullptr, 0);
        if (ret < 0) {
            return handle_error(ret, "Error initializing hardware device context");
        }
        hw_ctx.reset(tmp_hw_ctx);
    }

    // Initialize input decoder
    decoder::Decoder decoder;
    ret = decoder.init(hw_device_type_, hw_ctx.get(), in_fname);
    if (ret < 0) {
        return handle_error(ret, "Failed to initialize decoder");
    }

    AVFormatContext* ifmt_ctx = decoder.get_format_context();
    AVCodecContext* dec_ctx = decoder.get_codec_context();
    int in_vstream_idx = decoder.get_video_stream_index();

    // Create and initialize the appropriate filter
    // Parse comma-separated device indices
    std::vector<std::unique_ptr<processors::Processor>> processors;
    std::stringstream ss(vk_device_list_);
    std::string device_idx;
    
    while (std::getline(ss, device_idx, ',')) {
        // Convert string to integer device index
        uint32_t vk_device_idx = std::stoul(device_idx);
        
        // Create processor instance for this device
        video2x::logger()->info("BL: Creating processor for device {}", vk_device_idx);
        auto start = std::chrono::high_resolution_clock::now();
        
        auto processor = processors::ProcessorFactory::instance().create_processor(proc_cfg_, vk_device_idx);
        if (processor == nullptr) {
            return handle_error(-1, "Failed to create filter instance for device " + device_idx);
        }
        processors.push_back(std::move(processor));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        logger()->info("BL: Creating processor for device {} took {} ms", vk_device_idx, duration.count());
        // Sleep 50ms between processor creation
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (processors.empty()) {
        return handle_error(-1, "No valid processor instances created");
    }

    // Initialize output dimensions based on filter configuration
    int output_width = 0, output_height = 0;
    processors[0]->get_output_dimensions(
        proc_cfg_, dec_ctx->width, dec_ctx->height, output_width, output_height
    );
    if (output_width <= 0 || output_height <= 0) {
        return handle_error(-1, "Failed to determine the output dimensions");
    }

    // Initialize the encoder
    encoder::Encoder encoder;
    auto start = std::chrono::high_resolution_clock::now();
    ret = encoder.init(
        hw_ctx.get(),
        out_fname,
        ifmt_ctx,
        dec_ctx,
        enc_cfg_,
        output_width,
        output_height,
        proc_cfg_.frm_rate_mul,
        in_vstream_idx
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    logger()->info("BL: Initializing encoder took {} ms", duration.count());
    if (ret < 0) {
        return handle_error(ret, "Failed to initialize encoder");
    }

    // Initialize all processors
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> init_threads;
    std::vector<int> init_results(processors.size());

    for (size_t i = 0; i < processors.size(); i++) {
        init_threads.emplace_back([&, i]() {
            logger()->info("BL: Initializing filter for device {}", i);
            auto start = std::chrono::high_resolution_clock::now();
            init_results[i] = processors[i]->init(dec_ctx, encoder.get_encoder_context(), hw_ctx.get());
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            logger()->info("BL: Initializing processor {} took {} ms", i, duration.count());
        });
    }

    // Wait for all threads to complete
    for (auto& thread : init_threads) {
        thread.join();
    }

    // Check results
    for (size_t i = 0; i < init_results.size(); i++) {
        if (init_results[i] < 0) {
            return handle_error(init_results[i], "Failed to initialize filter");
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    logger()->info("BL: Total processor initialization took {} ms", total_duration.count());

    // Process frames using the encoder and decoder
    ret = process_frames(decoder, encoder, processors);
    if (ret < 0) {
        return handle_error(ret, "Error processing frames");
    }

    // Write the output file trailer
    ret = av_write_trailer(encoder.get_format_context());
    if (ret < 0) {
        return handle_error(ret, "Error writing output file trailer");
    }

    // Check if an error occurred during processing
    if (ret < 0 && ret != AVERROR_EOF) {
        return handle_error(ret, "Error occurred");
    }

    // Processing has completed successfully
    state_.store(VideoProcessorMultiDeviceState::Completed);
    return 0;
}

// Process frames using the selected filter.
int VideoProcessorMultiDevice::process_frames(
    decoder::Decoder& decoder,
    encoder::Encoder& encoder,
    std::vector<std::unique_ptr<processors::Processor>>& processors
) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    // Get required objects
    AVFormatContext* ifmt_ctx = decoder.get_format_context();
    AVCodecContext* dec_ctx = decoder.get_codec_context();
    int in_vstream_idx = decoder.get_video_stream_index();
    AVFormatContext* ofmt_ctx = encoder.get_format_context();
    AVCodecContext* enc_ctx = encoder.get_encoder_context();
    int* stream_map = encoder.get_stream_map();

    // Reference to the previous frame does not require allocation
    // It will be cloned from the current frame
    std::unique_ptr<AVFrame, decltype(&avutils::av_frame_deleter)> prev_frame(
        nullptr, &avutils::av_frame_deleter
    );

    // Allocate space for the decoded frames
    std::unique_ptr<AVFrame, decltype(&avutils::av_frame_deleter)> frame(
        av_frame_alloc(), &avutils::av_frame_deleter
    );
    if (frame == nullptr) {
        logger()->critical("Error allocating frame");
        return AVERROR(ENOMEM);
    }

    // Allocate space for the decoded packets
    std::unique_ptr<AVPacket, decltype(&avutils::av_packet_deleter)> packet(
        av_packet_alloc(), &avutils::av_packet_deleter
    );
    if (packet == nullptr) {
        logger()->critical("Error allocating packet");
        return AVERROR(ENOMEM);
    }

    // Set the total number of frames in the VideoProcessingContext
    logger()->info("Estimating the total number of frames to process");
    total_frames_ = avutils::get_video_frame_count(ifmt_ctx, in_vstream_idx);

    if (total_frames_ <= 0) {
        logger()->warn("Unable to determine the total number of frames");
        total_frames_ = 0;
    } else {
        logger()->info("{} frames to process", total_frames_.load());
    }


    // Create queues and threads for each processor
    std::vector<std::mutex> queue_mutexes(processors.size());
    std::vector<std::condition_variable> queue_cvs(processors.size());
    std::vector<std::queue<FrameInfo>> frame_queues(processors.size());
    std::vector<std::thread> processor_threads;
    std::atomic<bool> processing_complete = false;
    // Thread-safe output list for processed frames
    // Using list since it provides O(1) removal from any position
    std::mutex output_mutex;
    std::list<FrameInfo> output_frames;

    // Create and start processor threads
    for (size_t i = 0; i < processors.size(); i++) {
        processor_threads.emplace_back([&, i]() {
            // Queue to store frame indices for this processor thread
            std::queue<int64_t> frame_idx_queue;
            while (!processing_complete || !frame_queues[i].empty()) {
                // Check if processing is paused
                if (state_.load() == VideoProcessorMultiDeviceState::Paused) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                // Wait for frames using condition variable
                std::unique_lock<std::mutex> lock(queue_mutexes[i]);
                queue_cvs[i].wait(lock, [&]() {
                    return !frame_queues[i].empty() || (processing_complete && frame_queues[i].empty());
                });

                // Check if we should exit
                if (processing_complete && frame_queues[i].empty()) {
                    break;
                }

                // Get next frame from queue
                FrameInfo frame_info = frame_queues[i].front();
                //video2x::logger()->info("BL: Processing frame {} for device {}", frame_info.frame_idx, i);
                frame_queues[i].pop();
                lock.unlock();
                // Push frame index to queue for this processor thread
                frame_idx_queue.push(frame_info.frame_idx);
                // Process frame using filter
                AVFrame* proc_frame = nullptr;
                ret = process_filtering(processors[i], frame_info.frame, &proc_frame);
                //video2x::logger()->info("BL: Processed frame {} for device {}, ret: {}", frame_info.frame_idx, i, ret);
                if (ret < 0 && ret != AVERROR(EAGAIN)) {
                    logger()->critical("Error processing frame");
                    return ret;
                }

                if (proc_frame == nullptr) {
                    video2x::logger()->info("BL: Processed frame is nullptr for device {}, continuing", i);
                    continue;
                }

                // Pop frame index from queue and use it for processed frame
                int64_t current_frame_idx = frame_idx_queue.front();
                frame_idx_queue.pop();
                
                FrameInfo processed_frame_info;
                processed_frame_info.frame = proc_frame;
                processed_frame_info.frame_idx = current_frame_idx;

                std::unique_lock<std::mutex> output_lock(output_mutex);
                // video2x::logger()->info("BL: Pushing frame {} to output queue for device {}", current_frame_idx, i);
                output_frames.push_back(processed_frame_info);
                output_lock.unlock();
            }

            // Flush the processor
            std::vector<AVFrame*> raw_flushed_frames;
            ret = processors[i]->flush(raw_flushed_frames);
            if (ret < 0) {
                av_strerror(ret, errbuf, sizeof(errbuf));
                logger()->critical("Error flushing processor: {}", errbuf);
                return ret;
            }
            // Push flushed frames to output queue
            for (AVFrame* raw_frame : raw_flushed_frames) {
                // Pop frame index from queue and use it for flushed frame
                int64_t current_frame_idx = frame_idx_queue.front(); 
                frame_idx_queue.pop();

                FrameInfo flushed_frame_info;
                flushed_frame_info.frame = raw_frame;
                flushed_frame_info.frame_idx = current_frame_idx;

                std::unique_lock<std::mutex> output_lock(output_mutex);
                video2x::logger()->info("BL: Pushing flushed frame {} to output queue for device {}", current_frame_idx, i);
                output_frames.push_back(flushed_frame_info);
                output_lock.unlock();
            }
            return 0;
        });
    }

    // Create encoding thread
    std::thread encoding_thread([&]() {
        int next_frame_idx = 0;
        UnsharpFilter unsharp;
        bool filter_initialized = false;
        while (true) {
            if (state_.load() == VideoProcessorMultiDeviceState::Aborted) {
                break;
            }

            // Get next frame to encode
            std::unique_lock<std::mutex> output_lock(output_mutex);
            if (output_frames.empty()) {
                output_lock.unlock();
                // Check if all processing threads are done and output queue is empty
                bool all_threads_done = true;
                for (const auto& thread : processor_threads) {
                    if (thread.joinable()) {
                        all_threads_done = false;
                        break;
                    }
                }
                if (all_threads_done) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Get frame with frame_idx matching next_frame_idx
            auto next_frame_it = std::find_if(
                output_frames.begin(),
                output_frames.end(),
                [next_frame_idx](const FrameInfo& frame_item) {
                    //video2x::logger()->info("BL: Checking frame {} against {}", frame_item.frame_idx, next_frame_idx);
                    return frame_item.frame_idx == next_frame_idx;
                }
            );

            // Skip if frame not found
            if (next_frame_it == output_frames.end()) {
                //video2x::logger()->info("BL: Frame not found, sleeping for 10ms");
                output_lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            //video2x::logger()->info("BL: Frame {} found, erasing and writing", next_frame_idx);
            FrameInfo frame_info = *next_frame_it;
            output_frames.erase(next_frame_it);
            output_lock.unlock();
            if (!filter_initialized) {
                std::string option = filter_options_.empty() ? "5:5:1.5" : filter_options_.c_str();
                video2x::logger()->info("BL: Initializing unsharp filter with options: {}", option);
                int ret = unsharp.init(
                    (AVPixelFormat)frame_info.frame->format,
                    frame_info.frame->width,
                    frame_info.frame->height,
                    enc_ctx->time_base,
                    option.c_str()
                );
                if (ret < 0) {
                    logger()->critical("Failed to initialize unsharp filter");
                    av_frame_free(&frame_info.frame);
                    break;
                }
                filter_initialized = true;
            }
            AVFrame* filtered_frame = av_frame_alloc();
            int ret = unsharp.filter_frame(frame_info.frame, filtered_frame);
            if (ret < 0) {
                logger()->critical("Failed to apply unsharp filter");
                av_frame_free(&frame_info.frame);
                av_frame_free(&filtered_frame);
                break;
            }
            ret = write_frame(filtered_frame, encoder, frame_info.frame_idx);
            if (ret < 0) {
                state_.store(VideoProcessorMultiDeviceState::Failed);
                av_frame_free(&frame_info.frame);
                av_frame_free(&filtered_frame);
                break;
            }
            logger()->debug(
                "Encoded frame {}/{}", 
                frame_info.frame_idx,
                total_frames_.load()
            );
            next_frame_idx++;
            av_frame_free(&frame_info.frame);
            av_frame_free(&filtered_frame);
        }
    });

    // Read frames from the input file
    while (state_.load() != VideoProcessorMultiDeviceState::Aborted) {
        // Check if any queue has less than 3 items
        bool queue_ready = false;
        for (const auto& queue : frame_queues) {
            if (queue.size() < 3) {
                queue_ready = true;
                break;
            }
        }
        if (!queue_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        ret = av_read_frame(ifmt_ctx, packet.get());
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                logger()->info("Reached end of file");
                break;
            }
            av_strerror(ret, errbuf, sizeof(errbuf));
            logger()->critical("Error reading packet: {}", errbuf);
            return ret;
        }

        if (packet->stream_index == in_vstream_idx) {
            // Send the packet to the decoder for decoding
            ret = avcodec_send_packet(dec_ctx, packet.get());
            if (ret < 0) {
                av_strerror(ret, errbuf, sizeof(errbuf));
                logger()->critical("Error sending packet to decoder: {}", errbuf);
                return ret;
            }

            // Process frames decoded from the packet
            while (state_.load() != VideoProcessorMultiDeviceState::Aborted) {
                // Sleep for 100 ms if processing is paused
                if (state_.load() == VideoProcessorMultiDeviceState::Paused) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                // Receive the decoded frame from the decoder
                ret = avcodec_receive_frame(dec_ctx, frame.get());
                if (ret == AVERROR(EAGAIN)) {
                    // No more frames from this packet
                    break;
                } else if (ret < 0) {
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    logger()->critical("Error decoding video frame: {}", errbuf);
                    return ret;
                }

                // Calculate this frame's presentation timestamp (PTS)
                frame->pts =
                    av_rescale_q(frame_idx_, av_inv_q(enc_ctx->framerate), enc_ctx->time_base);

                
                // Send the frame to the processors
                // Find queue with minimum size
                size_t min_queue_idx = 0;
                size_t min_size = std::numeric_limits<size_t>::max();
                
                for (size_t i = 0; i < frame_queues.size(); i++) {
                    std::unique_lock<std::mutex> lock(queue_mutexes[i]);
                    size_t queue_size = frame_queues[i].size();
                    if (queue_size < min_size) {
                        min_size = queue_size;
                        min_queue_idx = i;
                    }
                }
                //video2x::logger()->info("BL: Pushing frame {} to queue {} with size {}", frame_idx_.load(), min_queue_idx, min_size);

                // Push frame to queue with minimum size
                // We need to clone the frame because:
                // 1. The original frame (frame.get()) will be reused in the next iteration of the decoding loop
                // 2. Multiple processor threads need their own copy of the frame to process independently
                // 3. The original frame's memory will be freed/overwritten when we read the next frame
                {
                    std::unique_lock<std::mutex> lock(queue_mutexes[min_queue_idx]);
                    FrameInfo info;
                    info.frame = av_frame_clone(frame.get());
                    info.frame_idx = frame_idx_.load();
                    frame_queues[min_queue_idx].push(info);
                }
                queue_cvs[min_queue_idx].notify_one();
                av_frame_unref(frame.get());
                frame_idx_.fetch_add(1);
                logger()->debug("Processed frame {}/{}", frame_idx_.load(), total_frames_.load());
            }
        } else if (enc_cfg_.copy_streams && stream_map[packet->stream_index] >= 0) {
            ret = write_raw_packet(packet.get(), ifmt_ctx, ofmt_ctx, stream_map);
            if (ret < 0) {
                return ret;
            }
        }
        av_packet_unref(packet.get());
    }

    avcodec_send_packet(dec_ctx, nullptr);
    // Process remaining frames (e.g., last 1–2 frames)

    while (avcodec_receive_frame(dec_ctx, frame.get()) == 0) {
        video2x::logger()->info("BL: Processing remaining frame {}", frame_idx_.load());
        // Calculate this frame's presentation timestamp (PTS)
        frame->pts =
            av_rescale_q(frame_idx_, av_inv_q(enc_ctx->framerate), enc_ctx->time_base);
        // Send the frame to the processors
        // Find queue with minimum size
        size_t min_queue_idx = 0;
        size_t min_size = std::numeric_limits<size_t>::max();
        
        for (size_t i = 0; i < frame_queues.size(); i++) {
            std::unique_lock<std::mutex> lock(queue_mutexes[i]);
            size_t queue_size = frame_queues[i].size();
            if (queue_size < min_size) {
                min_size = queue_size;
                min_queue_idx = i;
            }
        }
        video2x::logger()->info("BL: Pushing frame {} to queue {} with size {}", frame_idx_.load(), min_queue_idx, min_size);
        // Push frame to queue with minimum size
        {
            std::unique_lock<std::mutex> lock(queue_mutexes[min_queue_idx]);
            FrameInfo info;
            info.frame = av_frame_clone(frame.get());
            info.frame_idx = frame_idx_.load();
            frame_queues[min_queue_idx].push(info);
        }
        queue_cvs[min_queue_idx].notify_one();
        av_frame_unref(frame.get());
        frame_idx_.fetch_add(1);
        logger()->info("Processed frame {}/{}", frame_idx_.load(), total_frames_.load());
    }

    // Notify all processor threads to finish
    processing_complete = true;
    for (auto& cv : queue_cvs) {
        cv.notify_all();
    }

    // Wait for all processor threads to finish
    for (auto& thread : processor_threads) {
        thread.join();
    }

    // Wait for encoding thread to finish
    encoding_thread.join();

    // Flush the encoder    
    ret = encoder.flush();
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        logger()->critical("Error flushing encoder: {}", errbuf);
        return ret;
    }

    return ret;
}

int VideoProcessorMultiDevice::write_frame(AVFrame* frame, encoder::Encoder& encoder, int64_t frame_idx) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    std::unique_lock<std::mutex> lock(encoder_mutex_);

    if (!benchmark_) {
        ret = encoder.write_frame(frame, frame_idx);
        if (ret < 0) {
            av_strerror(ret, errbuf, sizeof(errbuf));
            logger()->critical("Error encoding/writing frame: {}", errbuf);
        }
    }
    return ret;
}

int VideoProcessorMultiDevice::write_raw_packet(
    AVPacket* packet,
    AVFormatContext* ifmt_ctx,
    AVFormatContext* ofmt_ctx,
    int* stream_map
) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;
    std::unique_lock<std::mutex> lock(encoder_mutex_);
    AVStream* in_stream = ifmt_ctx->streams[packet->stream_index];
    int out_stream_idx = stream_map[packet->stream_index];
    AVStream* out_stream = ofmt_ctx->streams[out_stream_idx];

    av_packet_rescale_ts(packet, in_stream->time_base, out_stream->time_base);
    packet->stream_index = out_stream_idx;

    ret = av_interleaved_write_frame(ofmt_ctx, packet);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        logger()->critical("Error muxing audio/subtitle packet: {}", errbuf);
    }
    return ret;
}

int VideoProcessorMultiDevice::process_filtering(
    std::unique_ptr<processors::Processor>& processor,
    AVFrame* frame,
    AVFrame** out_frame
) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    // Cast the processor to a Filter
    processors::Filter* filter = static_cast<processors::Filter*>(processor.get());

    // Process the frame using the filter
    ret = filter->filter(frame, out_frame);

    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        logger()->critical("Error filtering frame: {}", errbuf);
    }
    return ret;
}

}  // namespace video2x
