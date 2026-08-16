# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.46.0.dev0"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "configuration_utils": ["PretrainedConfig"],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_dvclive_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Losses
    # Models
    "models": [],
    "models.auto": [
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    "models.chameleon": [
        "ChameleonConfig",
        "ChameleonProcessor",
        "ChameleonVQVAEConfig",
    ],
    "models.llama": [],
    "processing_utils": ["ProcessorMixin"],
    "quantizers": [],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": [
        "EvalPrediction",
        "IntervalStrategy",
        "SchedulerType",
        "enable_full_determinism",
        "set_seed",
    ],
    "training_args": ["TrainingArguments"],
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
    "utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "is_apex_available",
        "is_av_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_sacremoses_available",
        "is_safetensors_available",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_speech_available",
        "is_tensorflow_text_available",
        "is_tf_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torch_tpu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AwqConfig",
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "GPTQConfig",
        "HqqConfig",
        "QuantoConfig",
        "TorchAoConfig",
    ],
}

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.llama"].append("LlamaTokenizer")

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Tensorflow-text-specific objects

# keras-nlp-specific objects

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.chameleon"].append("ChameleonImageProcessor")

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["activations"] = []
    _import_structure["cache_utils"] = [
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EosTokenCriteria",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "LogitsWarper",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "StopStringCriteria",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    # PyTorch domain libraries integration
    _import_structure["integrations.executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS"]
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch models structure


    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
            "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_CTC_MAPPING",
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
            "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_IMAGE_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
            "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_VISION_2_SEQ_MAPPING",
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoModel",
            "AutoModelForAudioClassification",
            "AutoModelForAudioFrameClassification",
            "AutoModelForAudioXVector",
            "AutoModelForCausalLM",
            "AutoModelForCTC",
            "AutoModelForDepthEstimation",
            "AutoModelForDocumentQuestionAnswering",
            "AutoModelForImageClassification",
            "AutoModelForImageSegmentation",
            "AutoModelForImageTextToText",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForKeypointDetection",
            "AutoModelForMaskedImageModeling",
            "AutoModelForMaskedLM",
            "AutoModelForMaskGeneration",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForObjectDetection",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSemanticSegmentation",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTextEncoding",
            "AutoModelForTextToSpectrogram",
            "AutoModelForTextToWaveform",
            "AutoModelForTokenClassification",
            "AutoModelForUniversalSegmentation",
            "AutoModelForVideoClassification",
            "AutoModelForVision2Seq",
            "AutoModelForVisualQuestionAnswering",
            "AutoModelForZeroShotImageClassification",
            "AutoModelForZeroShotObjectDetection",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.chameleon"].extend(
        [
            "ChameleonForConditionalGeneration",
            "MMoEChameleonForConditionalGeneration",
            "ChameleonModel",
            "ChameleonPreTrainedModel",
            "ChameleonProcessor",
            "ChameleonVQVAE",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
        "get_wsd_schedule",
    ]
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]

# TensorFlow-backed objects

# FLAX-backed objects

# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    # Agents
    from .configuration_utils import PretrainedConfig

    # Data
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    from .generation import GenerationConfig, TextIteratorStreamer, TextStreamer, WatermarkingConfig
    from .hf_argparser import HfArgumentParser

    # Integrations
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

    # Model Cards
    from .modelcard import ModelCard

    # TF 2.0 <=> PyTorch conversion utilities
    from .models.auto import (
        CONFIG_MAPPING,
        FEATURE_EXTRACTOR_MAPPING,
        IMAGE_PROCESSOR_MAPPING,
        MODEL_NAMES_MAPPING,
        PROCESSOR_MAPPING,
        TOKENIZER_MAPPING,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )
    from .models.chameleon import (
        ChameleonConfig,
        ChameleonProcessor,
        ChameleonVQVAEConfig,
    )

    # Pipelines
    from .processing_utils import ProcessorMixin

    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Trainer
    from .trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from .trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
    from .training_args import TrainingArguments
    from .training_args_seq2seq import Seq2SeqTrainingArguments

    # Files and general utilities
    from .utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_av_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_sacremoses_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from .utils.quantization_config import (
        AqlmConfig,
        AwqConfig,
        BitNetConfig,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
        EetqConfig,
        FbgemmFp8Config,
        GPTQConfig,
        HqqConfig,
        QuantoConfig,
        TorchAoConfig,
    )

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_sentencepiece_objects import *
    else:
        from .models.llama import LlamaTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tokenizers_objects import *
    else:
        # Fast tokenizers imports
        from .models.llama import LlamaTokenizerFast
        from .tokenization_utils_fast import PreTrainedTokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        from .convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_vision_objects import *
    else:
        from .image_processing_base import ImageProcessingMixin
        from .image_processing_utils import BaseImageProcessor
        from .image_utils import ImageFeatureExtractionMixin
        from .models.chameleon import ChameleonImageProcessor

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:
        # Benchmarks
        from .cache_utils import (
            Cache,
            CacheConfig,
            DynamicCache,
            EncoderDecoderCache,
            HQQQuantizedCache,
            HybridCache,
            MambaCache,
            OffloadedCache,
            OffloadedStaticCache,
            QuantizedCache,
            QuantizedCacheConfig,
            QuantoQuantizedCache,
            SinkCache,
            SlidingWindowCache,
            StaticCache,
        )
        from .generation import (
            AlternatingCodebooksLogitsProcessor,
            BeamScorer,
            BeamSearchScorer,
            ClassifierFreeGuidanceLogitsProcessor,
            ConstrainedBeamSearchScorer,
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EosTokenCriteria,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            GenerationMixin,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MaxLengthCriteria,
            MaxTimeCriteria,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PhrasalConstraint,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkDetector,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )
        from .modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from .modeling_utils import PreTrainedModel
        from .models.auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_XVECTOR_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
            MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
            MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
            MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
            MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
            MODEL_FOR_MASK_GENERATION_MAPPING,
            MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TEXT_ENCODING_MAPPING,
            MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
            MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
            MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
            MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForAudioFrameClassification,
            AutoModelForAudioXVector,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForDepthEstimation,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageTextToText,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForKeypointDetection,
            AutoModelForMaskedImageModeling,
            AutoModelForMaskedLM,
            AutoModelForMaskGeneration,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTextEncoding,
            AutoModelForTextToSpectrogram,
            AutoModelForTextToWaveform,
            AutoModelForTokenClassification,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForVision2Seq,
            AutoModelForVisualQuestionAnswering,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
            AutoModelWithLMHead,
        )
        from .models.chameleon import (
            ChameleonForConditionalGeneration,
            MMoEChameleonForConditionalGeneration,
            ChameleonModel,
            ChameleonPreTrainedModel,
            ChameleonProcessor,
            ChameleonVQVAE,
        )

        # PyTorch model imports

        # Optimization
        from .optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_inverse_sqrt_schedule,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
            get_wsd_schedule,
        )
        from .pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

        # Trainer
        from .trainer import Trainer
        from .trainer_pt_utils import torch_distributed_zero_first
        from .trainer_seq2seq import Seq2SeqTrainer

    # TensorFlow

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
