import torch
import timm
import time
import pandas as pd
import numpy as np

models = timm.list_models()
device = "cuda"
N = 50   # Number of evaluation to mean

for batch_size in [8,16,32,64]:
    timing = []
    for k, m in enumerate(models):
        model = timm.create_model(m, pretrained=False)
        model.to(device)
        model.eval()

        input_size = model.default_cfg["input_size"]

        with torch.no_grad():
            try:
                elapsed_list = []
                # warmup
                image = torch.rand((batch_size, 3, input_size[1], input_size[2]), device=device)
                output = model(image)
                # Evaluation
                for i in range(N):
                    image = torch.rand((batch_size, 3, input_size[1], input_size[2]), device=device)
                    t = time.time()
                    output = model(image)
                    elapsed_list.append(time.time()-t)
                
            except RuntimeError:
                print(f"{m} : OUT OF MEMORY")
                timing.append((m, -1, input_size[1]))
                continue
            except ImportError:
                # Drop ABN optimization
                print("skipping ", m)
                continue

            elapsed = np.mean(elapsed_list)*1000
            print(f"{m} : {elapsed:.4f} ms")
            timing.append((m, elapsed, input_size[1]))


    timing_df = pd.DataFrame()
    model_list, inf_time_list, input_size_list = zip(*timing)
    timing_df["model"] = list(model_list)
    timing_df["inf_time"] = list(inf_time_list)
    timing_df["input_size"] = list(input_size_list)
    timing_df.to_csv(f"timing_{batch_size}.csv", index=False)
