# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, get_file_lists
from nemo.utils import logging

"""
Currently Supported ASR models:

QuartzNet15x5Base

"""

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/speaker_diarization.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--audiofile_list_path", type=str, help="Fullpath of a file contains the list of audio files", required=True)
parser.add_argument("--reference_rttmfile_list_path", default=None, type=str, help="Fullpath of a file contains the list of rttm files")
parser.add_argument("--output_path", default=None, type=str, help="Path to the folder where output files are generated")
parser.add_argument("--pretrained_vad_model", default=None, type=str, help="Fullpath of the VAD model (*.nemo).")
parser.add_argument("--external_vad_manifest", default=None, type=str, help="External VAD file for diarization")
parser.add_argument('--asr_based_vad', default=False, action='store_true', help="Whether to use ASR-based VAD")
parser.add_argument('--generate_oracle_manifest', default=False, action='store_true', help="Whether to generate and use oracle VAD")
parser.add_argument("--pretrained_speaker_model", type=str, help="Fullpath of the Speaker embedding extractor model (*.nemo).", required=True,)
parser.add_argument("--oracle_num_speakers", help="Either int or text file that contains number of speakers")
parser.add_argument("--threshold", default=50, type=int, help="Threshold for ASR based VAD")
parser.add_argument("--diar_config_url", default=CONFIG_URL, type=str, help="Config yaml file for running speaker diarization")
parser.add_argument("--csv", default='result.csv', type=str, help="")
args = parser.parse_args()


vad_choices = [args.asr_based_vad, args.pretrained_vad_model, args.external_vad_manifest, args.generate_oracle_manifest]
if not sum(bool(c) for c in vad_choices) == 1:
    raise ValueError("Please either provide ONE and ONLY ONE method for VAD. \n \
            (1) External manifest external_vad_manifest \n \
            (2) Use a pretrained_vad_model \n \
            (3) Use ASR-based VAD or \n \
            (4) Provide reference_rttmfile_list_path so we can automatically generate oracle manifest for diarization"
        )
    if args.generate_oracle_manifest and not args.reference_rttmfile_list_path:
        raise ValueError("Please provide reference_rttmfile_list_path so we can generate oracle manifest automatically") 


if args.asr_based_vad:
    oracle_manifest = 'asr_based_vad'
elif args.pretrained_vad_model:  
    oracle_manifest = 'system_vad'
elif args.external_vad_manifest:
    oracle_manifest = args.external_vad_manifest
elif args.reference_rttmfile_list_path:
    logging.info("Use the oracle manifest automatically generated by rttm")
    oracle_manifest = asr_diar_offline.write_VAD_rttm(asr_diar_offline.oracle_vad_dir, audio_file_list, args.reference_rttmfile_list_path)

params = {
    "round_float": 2,
    "window_length_in_sec": 1.5,
    "shift_length_in_sec": 0.75,
    "print_transcript": False,
    "lenient_overlap_WDER": True,
    "fix_word_ts_with_SAD": False,
    "SAD_threshold_for_word_ts": 0.7,
    "max_word_ts_length_in_sec": 0.6,
    "word_gap_in_sec": 0.01,
    "minimum": True,
    "threshold": args.threshold,  # minimun width to consider non-speech activity
    "asr_based_vad": args.asr_based_vad,
    "diar_config_url": args.diar_config_url,
    "ASR_model_name": 'stt_en_conformer_ctc_large',
    # "ASR_model_name": 'QuartzNet15x5Base-En', 
}

asr_diar_offline = ASR_DIAR_OFFLINE(params)

asr_model = asr_diar_offline.set_asr_model(params['ASR_model_name'])

asr_diar_offline.create_directories(args.output_path)

audio_file_list = get_file_lists(args.audiofile_list_path)

word_list, word_ts_list = asr_diar_offline.run_ASR(asr_model, audio_file_list)

diar_labels = asr_diar_offline.run_diarization(
    audio_file_list, 
    word_ts_list,
    oracle_manifest=oracle_manifest, 
    oracle_num_speakers=args.oracle_num_speakers, 
    pretrained_speaker_model=args.pretrained_speaker_model,
    pretrained_vad_model=args.pretrained_vad_model
)

if args.reference_rttmfile_list_path:

    ref_rttm_file_list = get_file_lists(args.reference_rttmfile_list_path)

    ref_labels_list, DER_result_dict = asr_diar_offline.eval_diarization(
        audio_file_list, diar_labels, ref_rttm_file_list
    )
    
    total_riva_dict = asr_diar_offline.write_json_and_transcript(audio_file_list, diar_labels, word_list, word_ts_list)
   
    WDER_dict = asr_diar_offline.get_WDER(audio_file_list, total_riva_dict, DER_result_dict, ref_labels_list)
    
    effective_wder = asr_diar_offline.get_effective_WDER(DER_result_dict, WDER_dict)
    # print(effective_wder)
    logging.info(
        f"\nDER  : {DER_result_dict['total']['DER']:.4f} \
          \nFA   : {DER_result_dict['total']['FA']:.4f} \
          \nMISS : {DER_result_dict['total']['MISS']:.4f} \
          \nCER  : {DER_result_dict['total']['CER']:.4f} \
          \nWDER : {WDER_dict['total']:.4f} \
          \neffective WDER : {effective_wder:.4f} \
          \nspk_counting_acc : {DER_result_dict['total']['spk_counting_acc']:.4f}"
    )

else:
    diar_labels = asr_diar_offline.get_diarization_labels(audio_file_list)
    
    total_riva_dict = asr_diar_offline.write_json_and_transcript(audio_file_list, diar_labels, word_list, word_ts_list)

