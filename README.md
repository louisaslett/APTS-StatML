# APTS Statistical Machine Learning module

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/louisaslett/APTS-StatML?quickstart=1)

The [Academy of PhD Training in Statistics (APTS)](https://warwick.ac.uk/fac/sci/statistics/apts/) is a collaboration between major UK statistics research groups to organise courses for first-year PhD students in statistics and applied probability nationally.
This repository provides a [Github Codespaces](https://github.com/features/codespaces/) for running the two computer labs in the [week 4 module](https://warwick.ac.uk/fac/sci/statistics/apts/programme/) on Statistical Machine Learning, delivered by [Louis Aslett](https://www.louisaslett.com/).

## Links

- [Setup Instructions](https://www.louisaslett.com/StatML/setup/)
  - This explains how to create a Github account and to register your student account in order to get free "Pro" benefits, such as extra [Github Codespaces](https://github.com/features/codespaces/) allowance.
    Please complete this before the first lab.
- [Preliminary Material](https://www.louisaslett.com/StatML/prelim/)
  - Please study this material prior to the start of the course.
    It will help with getting a sense of the setting of the course.
- [Course Notes](https://www.louisaslett.com/StatML/notes/)
  - These are the main notes for the course which we will work through in the lectures during the week, scheduled to run from 9th-13th September 2024 at the University of Oxford.
- [Labs](https://www.louisaslett.com/StatML/notes/ml-pipelines.html)
  - Each lab is about 1.5 hours and there will be help on hand if you get stuck.
    At the link above you will find the question sheet, and a full solution web page.
    In addition, the code (without solution) is extracted into R files to save you copy-and-pasting.
    The same two R files `lab1.R` and `lab2.R` are in this repository and will be visible in the Codespace when you launch it.

## Labs

There are two labs during the course which make use of [R](https://www.r-project.org/) and various packages, mainly run via either [Tidymodels](https://www.tidymodels.org/) or [MLR3](https://mlr3.mlr-org.com/).
To simplify the running of the labs and ensure a consistent software environment, it is recommended to use the devcontainer provided in this repository via [Github Codespaces](https://github.com/features/codespaces) ... note this is free to use if you associate your `.ac.uk` academic email address with your Github account and [register as a student](https://education.github.com/pack).
Please follow the ["Setup Instructions"](https://www.louisaslett.com/StatML/setup/) linked above before the labs so that you're ready to use Github.
However, you can also run R locally if you prefer, or attempt to use the University lab computers (no guarantees they'll be up-to-date!)

## Module Description

**Aims:**
This module introduces students to modern supervised machine learning methodology and practice, with an emphasis on statistical and probabilistic approaches in the field.
The course seeks to balance theory, methods and application, providing an introduction with firm foundations that is accessible to those working on applications and seeking to employ best practice.
There will be exploration of some key software tools which have facilitated the growth in the use of these methods across a broad spectrum of applications and an emphasis on how to carefully assess machine learning models.

**Learning Outcomes:**
Students following this module will gain a broad view of the supervised statistical machine learning landscape, including some of the main theoretical and methodological foundations.
They will be able to appraise different machine learning methods and reason about their use.
In particular, students completing the course will gain an understanding of how to practically apply these techniques, with an ability to critically evaluate the performance of their models.
Students will also have an insight into the extensive software libraries available today and their use to construct a full machine learning pipeline.

**Prerequisites**

To undertake this module, students should have:

- at least one undergraduate level course in probability and in statistics;
- standard undergraduate level knowledge of linear algebra and calculus;
- solid grasp of statistical computing in R;
- knowledge of statistical modelling, including regression modelling (eg. APTS Statistical Modelling course);
- some basic understanding of optimisation methods beneficial, but not essential.

As preparatory reading, the enthusiastic student may choose to browse An Introduction to Statistical Learning (James et al., 2013) ([freely and legally available online here](https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf)), which covers some of the topics of the course at a more elementary and descriptive level.

Textbooks at roughly the level of the course include:

- The Elements of Statistical Learning (Friedman, Tibshirani, and Hastie)
- Pattern Recognition and Machine Learning (Bishop)
- Machine Learning: A Probabilistic Perspective (Murphy)

**Topics**

- Formulation of supervised learning for regression and classification (scoring/probabilistic, decision boundaries, generative/discriminative), loss functions and basic decision theory;
- Theory of model capacity, complexity and bias-variance decomposition;
- Curse of dimensionality;
- Overview of some key modelling methodologies (eg logistic regression, local methods, kernel smoothing, trees, boosting, bagging, forests);
- Model selection, ensembles, tuning and super-learning;
- Evaluation of model performance, validation and calibration and their
- reporting in applications;
- Reproducibility;
- Coverage of some key software frameworks for applying machine learning in the real world.

**Assessment:**
An exercise set by the module leader involving practical use of some of the machine learning methods covered and critical evaluation of their performance.
