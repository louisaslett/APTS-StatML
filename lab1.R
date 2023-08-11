# Uncomment and run the following command first if you do not have the modeldata package
# install.packages("modeldata")
data("credit_data", package = "modeldata")

# Try some plotting function to explore the data here


# Uncomment and run the following command first if you do not have the fst package
# install.packages("fst")
download.file("https://www.louisaslett.com/Courses/MISCADA/mnist.fst", "mnist.fst")

mnist <- fst::read.fst("mnist.fst")

# Define a plotting function to transform the flattened image data into a
# square graphic. See Data Coding appendix in the notes for how images are
# represented as tabular data!
library("tidyverse")

plotimages <- function(i) {
  imgs <- mnist |>
    slice(i) |>
    mutate(image.num = i) |>
    pivot_longer(x0.y27:x27.y0,
                 names_to = c("x", "y"),
                 names_pattern = "x([0-9]{1,2}).y([0-9]{1,2})",
                 names_transform = list(x = as.integer, y = as.integer),
                 values_to = "greyscale")

  ggplot(imgs, aes(x = x, y = y, fill = greyscale)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    facet_wrap(~ image.num + response, labeller = label_both)
}

plotimages(21:24)

# Uncomment and run the following command first if you do not have the tidymodels package
# install.packages("tidymodels")

# Load tidymodels, which loads a suite of other packages for us
library("tidymodels")

# Set seed so we reproducibly see the same results
set.seed(212)

# Because knn can't handle missings ... maybe try imputation later?
credit_data <- na.omit(credit_data)

# Create a train/test split into 75% / 25% of the data
credit_data_split <- initial_split(credit_data, strata = Status, prop = 3/4)

# We then extract the training and testing portions
train_data <- training(credit_data_split)
test_data <- testing(credit_data_split)

# First, define a logistic regression model
lr_mod <- logistic_reg()


# Now we need to load the kknn package due to a bug waiting a fix
library("kknn") # Needed due to bug! https://github.com/tidymodels/parsnip/issues/264

# Second, define a knn model. Since KNN has a hyperparameter we can specify that
# here, and because it supports both classification and regression we can
# set which mode we want
knn_mod <- nearest_neighbor(neighbors = 3) |>
  set_mode("classification")

# Next we fit the model
lr_fit  <- lr_mod  |> fit(Status ~ ., data = train_data)
knn_fit <- knn_mod |> fit(Status ~ ., data = train_data)

lr_fit
knn_fit

# APPARENT ERRORS!

lr_apparent <-
  # First column will be predicted label
  predict(lr_fit, train_data) |>
  # Next two columns will be probabilities for the two labels
  bind_cols(predict(lr_fit, train_data, type = "prob")) |>
  # Last column will be true label
  bind_cols(train_data |> select(Status))
head(lr_apparent)

knn_apparent <-
  # First column will be predicted label
  predict(knn_fit, train_data) |>
  # Next two columns will be probabilities for the two labels
  bind_cols(predict(knn_fit, train_data, type = "prob")) |>
  # Last column will be true label
  bind_cols(train_data |> select(Status))
head(knn_apparent)

mn_log_loss(lr_apparent, truth = Status, .pred_bad)
mn_log_loss(knn_apparent, truth = Status, .pred_bad)

accuracy(lr_apparent, truth = Status, .pred_class)
accuracy(knn_apparent, truth = Status, .pred_class)

conf_mat(lr_apparent, truth = Status, .pred_class)
conf_mat(knn_apparent, truth = Status, .pred_class)

roc_auc(lr_apparent, truth = Status, .pred_bad)
roc_auc(knn_apparent, truth = Status, .pred_bad)

# TEST ERRORS!
# Adapt the code above to compute the same measures on the test data


# CROSS VALIDATION!
# We'll now do 3-fold cross validation (perhaps want to do more, we choose 3
# just to make sure the lab code runs quite quickly)
credit_data_cv <- vfold_cv(train_data, v = 3)

split1_train <- training(credit_data_cv$splits[[1]])
split1_test <- testing(credit_data_cv$splits[[1]])

split2_train <- training(credit_data_cv$splits[[2]])
split2_test <- testing(credit_data_cv$splits[[2]])

split3_train <- training(credit_data_cv$splits[[3]])
split3_test <- testing(credit_data_cv$splits[[3]])

# Better than loops is to use workflows in tidymodels
lr_wf <- workflow() |>
  add_model(lr_mod) |>
  add_formula(Status ~ .)

lr_cv_fits <- lr_wf |>
  fit_resamples(credit_data_cv)

lr_cv_fits |> collect_metrics()

# Update metrics below to compute with cross-entropy loss and 0-1 loss
lr_cv_fits <- lr_wf |>
  fit_resamples(credit_data_cv,
                metrics = ???) |>
  collect_metrics()
lr_cv_fits

# Uncomment and run the following command first if you do not have the glmnet package
# install.packages("glmnet")

# Adjust the mixture parameter below to perform L1 regularisation

lr_grid <- grid_regular(penalty(),
                        levels = 10)

lr_tune_mod <- logistic_reg(penalty = tune(),
                            mixture = ???) |>
  set_engine("glmnet")

# Now, use the metrics found before for tune_grid

lr_tune_wf <- workflow() |>
  add_model(lr_tune_mod) |>
  add_formula(Status ~ .)

lr_tune_cv_fits <- lr_tune_wf |>
  tune_grid(
    resamples = credit_data_cv,
    grid = lr_grid,
    metrics = ???
  )

lr_tune_cv_fits |> collect_metrics()

# Repeat, starting out with a different grid to zoom in on area of relevance
lr_grid <- grid_regular(penalty(???),
                        levels = 10)

# (rerun the previous solution you came up with after changing that line)

# This plot may help to see what's going on!
lr_tune_cv_fits |>
  collect_metrics() |>
  ggplot(aes(penalty, mean)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number())

# Try tuning the k-nearest neighbour to select number of neighbours


# Examine best model
lr_tune_cv_fits |> show_best("accuracy")
lr_tune_cv_fits |> show_best("mn_log_loss")

# Pull out best model
# Really, want to make sure you use a loss *relevant to your real problem* for
# final selection of model!
lr_final_mod <- lr_tune_cv_fits |> select_best("mn_log_loss")

# First, finalise the workflow to only the best model
lr_final_wf <- lr_tune_wf |>
  finalize_workflow(lr_final_mod)

# Then fit it ... note we now use the credit_data_split from way back at the start
# of the lab, so that we use all the training_data (which was split for CV) to
# fit and then test on the test data
lr_final_fit <-
  lr_final_wf |>
  last_fit(credit_data_split,
           metrics = ???)

lr_final_fit |> collect_metrics()

# Repeat for your KNN model!


