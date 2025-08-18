# SPDX-License-Identifier: Apache-2.0

import os
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

from vllm import LLM, SamplingParams


def test_qwen():
    llm = LLM(model="Qwen/Qwen3-0.6B-FP8",
              tensor_parallel_size=2,
              max_num_seqs=4,
              max_model_len=128,
              enforce_eager=True,
              override_neuron_config={
                  "sequence_parallel_enabled": False,
                  "skip_warmup": True,
              })

    prompts = [
        "The president of the United States is",
        "The capital of France is",
    ]
    outputs = llm.generate(prompts, SamplingParams(top_k=1))

    expected_outputs = [
        " the most powerful person in the world. He is the head of state "
        "and head",
        " a city of many faces. It is a city of history, culture, art"
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        assert (expected_output == generated_text)