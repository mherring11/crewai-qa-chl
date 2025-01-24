from crewai import Task
from textwrap import dedent


class QuestionAnalysisTasks:
    @staticmethod
    def simulate_answer(agent, question):
        return Task(
            description=dedent(f"Simulate an answer for the following question:\n\n{question}"),
            agent=agent
        )

    @staticmethod
    def evaluate_answer(agent, question, simulated_answer):
        return Task(
            description=dedent(f"Evaluate the simulated answer for the question:\n\n{question}\n\nAnswer:\n{simulated_answer}"),
            agent=agent
        )
