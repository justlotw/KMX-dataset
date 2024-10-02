# KMX Dataset
## Unraveling the Complexity of Math Problems with the KMX Dataset

Mathematical reasoning is a crucial aspect in evaluating human cognition and is a crucial part of many decision-making tasks. While current datasets focus on a narrow set of topics or question types, the KMX(Khan-Math-X) dataset is a comprehensive and diverse set of problems that is annotated with additional dimensions such as grade level, exercise names, and question types. This allows for more fine-grained analysis of model performance. We assess the performance of several models using various techniques and show that while certain methods improve performance across the board, their effectiveness varies significantly depending on the model and problem type, revealing the intricacies when dealing with different mathematical tasks. The KMX dataset enables a more nuanced understanding of model strengths and weaknesses, and offers a valuable resource for future research in mathematical reasoning.

<img align="center" src="dataset_comparison.png">

## Dataset Details
The KMX (Khan-Math-X) dataset adheres to some design principles that are aimed at targeting gaps in current datasets. 

- **Alignment with Educational Standards**: The KMX dataset is designed to comprehensively represent real-world math problems sourced from the Khan Academy curriculum. Each problem is carefully annotated with grade, unit, lesson, and topic to ensure educational relevance and to facilitate fine-grained performance evaluation across a broad range of math skills.
- **Diversity and Depth for Robust Evaluation**: To support in-depth analysis, the dataset encompasses a variety of question types, including word problems and mathematical expressions, categorized into 18 distinct topics, and 9 question types. We strive to have a large distribution of math topics and question types available so that a more holistic understanding of models' numerical reasoning capabilities can be accessed.
- **Smooth Progression for Model Evaluation**: The dataset is designed to accommodate a wide range of models, from the weakest to the most advanced. We aim for a smooth gradient in problem difficulty to ensure that high-performing models can excel, while lower-performing models still demonstrate meaningful progression. This approach provides researchers with a clear understanding of each model’s strengths and weaknesses, facilitating more precise benchmarking across different levels of model capability.
- **Ease of Use**: The KMX dataset is formatted to be readily usable by a wide range of models, with problems provided in a standardized, human-readable format. Additionally, each problem is accompanied by natural language, step-by-step solutions, making it easier for both models and researchers to interpret the reasoning behind the answers.

<center>

| Feature              | Value |
| :---------------- | :------: |
| Number of Problems        |   3240   |
| Unique Grades           |   8   |
| Unique Unit Names    |  34   |
| Unique Lesson Names    |  102   |
| Unique Exercise Names |  149   |
| Unique Topics           |   18   |
| Unique Question Types    |  9   |

</center>

The grade, unit names, lesson names and exercise names are sourced from Khan Academy. The topics and question types have been annotated to be aligned with the Math syllabus.

The data has been split into 2285 training problems and 955 test problems. They can be found in 
- [data/kmx_train.csv](data/kmx_train.csv)
- [data/kmx_test.csv](data/kmx_test.csv)


## Experiments
Sample scripts are provided in [scripts/](scripts/) to perform any experiments on the KMX dataset.

Relevant files:
- **finetune.py**: Performs fine-tuning on existing models. Accepts a [config file](scripts/finetune_config.json) for hyperparmeters.
- **inference.py**: Performs generation on a language model. Contains options to include various in-context-learning prompts.
- **inf_calc.py**: Performs generation on a language model that has been trained with the appropriate calculator tags
- **evaluate.py**: Parses the generated output and converts it into an answer that can be used to evaluate correctness

Full experimental results and discussions can be seen in the paper.

