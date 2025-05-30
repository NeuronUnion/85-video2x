import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    #in0 = torch.rand(1, 12, 64, 64, dtype=torch.float)
    in0 = torch.rand(1, 3, 64, 64, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("realesrgan-plus-x4.param")
        net.load_model("realesrgan-plus-x4.bin")

        with net.create_extractor() as ex:
            ex.input("data", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("output")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
