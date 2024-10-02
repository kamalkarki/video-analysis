# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.llama3.reference_impl.generation import Llama

tokenizer_path = '/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct/tokenizer.model'
ckpt_dir = "/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct"

def run_main(
    ckpt_dir: str =ckpt_dir,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: int = 64,
    model_parallel_size: Optional[int] = None,
):
    # print(ckpt_dir)    
    # tokenizer_path = str(tokenizer_path)
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    prompts = [
        "The color of the sky is blue but sometimes it can also be",
        """\
apple is pomme,
bannana is banane,
cherry is""",
        "1, 2, 3, 5, 8, 13",
        "ba ba black sheep, have you any wool?",
    ]
    for prompt in prompts:
        result = generator.text_completion(
            prompt,
            temperature=0.6,
            top_p=0.9,
            max_gen_len=max_gen_len,
            logprobs=False,
        )

        cprint(f"{prompt}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")


def main_test():
    fire.Fire(run_main)


main_test()

