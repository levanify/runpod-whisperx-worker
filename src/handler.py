import runpod
import whisperx
import os
import torch

HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda"
# device = "cpu"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
# compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)
model_dir = "./models"
model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=model_dir)
align_model, metadata = whisperx.load_align_model(language_code="en", device=device, model_dir=model_dir)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device, cache_dir=model_dir)


def handler(job):
    job_input = job['input']

    audio_file = job_input['audio_file']
    num_speakers = job_input['num_speakers']

    print("🕘Loading audio...")
    audio = whisperx.load_audio(audio_file)

    print("🕘Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    print("✅Transcription completed!")
    # print(result["segments"])  # before alignment

    print("🕘Aligning audio...")
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device, return_char_alignments=False
    )

    print("✅Alignment completed!")
    # print(result["segments"])  # after alignment

    print("🕘Diarizing audio...")

    diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)

    print("🕘Assigning word speakers...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print("✅Diarization completed!")
    # print(diarize_segments)

    final_result = ""
    for (i, seg) in enumerate(result["segments"]):
        temp_speaker = ""
        temp_sentence = ""
        temp_j = 0
        for (j, word) in enumerate(seg["words"]):
            if word.get("speaker") is None:
                word["speaker"] = temp_speaker

            if word["speaker"] == temp_speaker:
                temp_sentence += " " + word["word"]
            else:
                if temp_speaker != "":
                    final_result += f"Line {i}.{temp_j} - {temp_speaker}: {temp_sentence}\n"
                    temp_j += 1

                temp_speaker = word["speaker"]
                temp_sentence = word["word"]

        final_result += f"Line {i}.{temp_j} - {temp_speaker}: {temp_sentence}\n"

    print("✅Final result:")
    print(final_result)  # segments are now assigned speaker IDs

    return final_result


runpod.serverless.start({"handler": handler})
