import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import transformers.masking_utils as masking_utils


def patch_sdpa_mask_for_trace():
    """
    Work around a Transformers tracing bug where q_length can arrive as a
    scalar tensor and crash in sdpa_mask's legacy cache_position branch.
    """
    if getattr(masking_utils, "_trace_sdpa_patch_applied", False):
        return

    original_sdpa_mask = masking_utils.sdpa_mask

    def patched_sdpa_mask(*args, **kwargs):
        q_length = kwargs.get("q_length")
        q_offset = kwargs.get("q_offset")
        if isinstance(q_length, torch.Tensor) and q_length.dim() == 0:
            kwargs["q_length"] = int(q_length.item())
            if isinstance(q_offset, torch.Tensor):
                kwargs["q_offset"] = int(q_offset.item())
        return original_sdpa_mask(*args, **kwargs)

    masking_utils.sdpa_mask = patched_sdpa_mask
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = patched_sdpa_mask
    masking_utils._trace_sdpa_patch_applied = True

def export_model():
    print("Downloading and loading the model...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_sdpa_mask_for_trace()
    
    # --- THE FIX ---
    # Load the config first, and set the C++ compatibility flags explicitly
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True  # Forces C++ compatible operations
    config.return_dict = False # Forces standard tuple outputs
    # Newer versions of Transformers default to SDPA-based attention masking.
    # Force eager attention to avoid tracing failures in masking_utils.
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"

    # Load the model using our custom configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        attn_implementation="eager",
    )

    model.eval()

    # Create dummy inputs
    text = "This is a test sentence for the inference engine."
    inputs = tokenizer(text, return_tensors="pt")
    
    # We only need the input_ids and attention_mask
    dummy_input = (inputs['input_ids'], inputs['attention_mask'])

    print("Tracing the model into TorchScript format...")
    # Tracing records the operations
    traced_model = torch.jit.trace(model, dummy_input, strict=False)

    # Save the C++ compatible model
    save_path = "model.pt"
    traced_model.save(save_path)
    print(f"Success! Model exported to {save_path}")

if __name__ == "__main__":
    export_model()
