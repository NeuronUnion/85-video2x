#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "avutils.h"
#include "decoder.h"
#include "encoder.h"
#include "libvideo2x_export.h"
#include "processor.h"

namespace video2x {

enum class VideoProcessorMultiDeviceState {
    Idle,
    Running,
    Paused,
    Failed,
    Aborted,
    Completed
};

class LIBVIDEO2X_API VideoProcessorMultiDevice {
   public:
    VideoProcessorMultiDevice(
        const processors::ProcessorConfig proc_cfg,
        const encoder::EncoderConfig enc_cfg,
        const std::string &vk_device_list,
        const AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_NONE,
        const bool benchmark = false,
        const std::string& filter_options = ""
    );

    virtual ~VideoProcessorMultiDevice() = default;

    [[nodiscard]] int
    process(const std::filesystem::path in_fname, const std::filesystem::path out_fname);

    void pause() { state_.store(VideoProcessorMultiDeviceState::Paused); }
    void resume() { state_.store(VideoProcessorMultiDeviceState::Running); }
    void abort() { state_.store(VideoProcessorMultiDeviceState::Aborted); }

    VideoProcessorMultiDeviceState get_state() const { return state_.load(); }
    int64_t get_processed_frames() const { return frame_idx_.load(); }
    int64_t get_total_frames() const { return total_frames_.load(); }

   private:
    [[nodiscard]] int process_frames(
        decoder::Decoder& decoder,
        encoder::Encoder& encoder,
        std::vector<std::unique_ptr<processors::Processor>>& processors
    );

    [[nodiscard]] int write_frame(AVFrame* frame, encoder::Encoder& encoder, int64_t frame_idx);

    [[nodiscard]] inline int write_raw_packet(
        AVPacket* packet,
        AVFormatContext* ifmt_ctx,
        AVFormatContext* ofmt_ctx,
        int* stream_map
    );

    [[nodiscard]] inline int process_filtering(
        std::unique_ptr<processors::Processor>& processor,
        AVFrame* frame,
        AVFrame** out_frame
    );

    processors::ProcessorConfig proc_cfg_;
    encoder::EncoderConfig enc_cfg_;
    std::string vk_device_list_;
    AVHWDeviceType hw_device_type_ = AV_HWDEVICE_TYPE_NONE;
    bool benchmark_ = false;
    std::string filter_options_;
    std::atomic<VideoProcessorMultiDeviceState> state_ = VideoProcessorMultiDeviceState::Idle;
    std::atomic<int64_t> frame_idx_ = 0;
    std::atomic<int64_t> total_frames_ = 0;
    std::mutex encoder_mutex_;
};

}  // namespace video2x
