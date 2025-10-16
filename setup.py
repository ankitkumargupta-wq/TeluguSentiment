
from setuptools import setup, find_packages

setup(
    name="TeluguSentiment",
    version="0.2.0",
    author="Ankit Kumar",
    author_email="ankitkr3289@gmail.com",
    description="Telugu Sentiment Analysis Python package",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "transformers",
        "emoji",
        "pandas",
        "openpyxl"
    ],
)
