import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from deepkin.models.mamba import MambaConfig
from deepkin.models.mamba_py_asr import MambaMobileAsrModel
from deepkin.models.syllabe_vocab import BLANK_ID
from deepkin.models.syllabe_vocab import syllbe_vocab_size

if __name__ == '__main__':
    PRE_TRAINED_MAMBA_SSM_ASR_MODEL_FILE = "mamba_ssm_asr_model_base_2024-04-17.pt_best_valid_loss.pt"
    d_model, n_layers = 768, 20
    mamba_config = MambaConfig(d_model, n_layers)
    wrapper = MambaMobileAsrModel(mamba_config,syllbe_vocab_size(), BLANK_ID,
                                  pre_trained_mamba_asr_model = PRE_TRAINED_MAMBA_SSM_ASR_MODEL_FILE)

    wrapper = torch.jit.script(wrapper)

    # wrapper.save("scripted_wrapper_tuple.pt")
    #
    # wrapper = torch.jit.load("scripted_wrapper_tuple.pt")

    EXPORTED_FILE = "mamba_ssm_asr_model_base_2024-04-17_v2.0.ptl"

    scripted_model = torch.jit.script(wrapper)
    optimized_model = optimize_for_mobile(scripted_model, preserved_methods=['reset'])
    optimized_model._save_for_lite_interpreter(EXPORTED_FILE)

    print(f"Done! Generated: {EXPORTED_FILE}")
