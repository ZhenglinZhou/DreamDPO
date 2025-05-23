import re
import json


class PromptV1:
    def __init__(self):
        self.prompt = None  # prompt
        self.q_str = None  # yes or no questions based on the given prompt
        self.q_num = 0  # the number of question

        self.task_description = """You are an expert in evaluating the alignment between a given text description and an image. Your task is to answer each of the alignment questions with either "Yes" or "No," based on the image. Provide your responses in the format specified below."""

        self.eval_criteria = """
        Instructions:
        1. Carefully analyze the provided image and answer questions based on the image.
        2. For each question, answer with either "Yes" or "No". Do not provide explanations or additional information.
        """

    def init_question(self, prompt):
        data = json.load(open("verifiers/Qwen/question.json", 'r'))  # Load the prepared question json
        qs = data[prompt]  # find the questions based on the prompt
        q_str = "Evaluation Question(s):\n"
        for i, q in enumerate(qs):
            q_str += f'Q{i + 1}: {q}\n'

        # initialize prompt, question and question number
        self.prompt = prompt
        self.q_str = q_str
        self.q_num = (i + 1)
        print(f'Prompt: {prompt}\nNumber of question: {self.q_num}\nQuestions:{self.q_str}')

    def init_output(self):
        output_format = "Output Format:\n"
        for i in range(self.q_num):
            output_format += f'A{i + 1}: [Yes/No]\n'
        output_format += '...'
        self.output_format = output_format

    def extract_score(self, rsp):
        # Split the string into lines
        lines = rsp.split('\n')

        # Count the number of lines containing 'Yes'
        yes_count = sum(1 for line in lines if 'Yes' in line)
        return yes_count

    def get_prompt(self, prompt: str) -> (str):

        if self.prompt is None:
            # Initialize the prompt and questions in first time.
            self.init_question(prompt)
            self.init_output()

        question = self.q_str
        output_format = self.output_format

        return self.task_description + self.eval_criteria + question + output_format
