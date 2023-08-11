# Uncomment and run the following command first if you do not have the modeldata package
# install.packages("modeldata")
data("credit_data", package = "modeldata")

# Try some plotting function to explore the data here


# Uncomment and run the following command first if you do not have the mlr3 package
# install.packages("mlr3")

# Load mlr3, which loads a suite of other packages for us
library("mlr3")

# Define a task, which is a dataset together with target variable for prediction
# We wrap the data in an na.omit to avoid issues with missingness, see later for
# better options
task_credit <- TaskClassif$new(id = "credit",
                               backend = na.omit(credit_data),
                               target = "Status")
task_credit

# This variable shows available learning algorithms
as.data.table(mlr_learners)

# Load more learners in supporting packages
library("mlr3learners")

as.data.table(mlr_learners)

# ... whilst this just gives the names
mlr_learners

#  # Get more details on learners
#  View(as.data.table(mlr_learners))

# Define a logistic regression model
learner_lr <- lrn("classif.log_reg")
learner_lr

# Train the model
learner_lr$train(task_credit)

# Perform prediction
pred <- learner_lr$predict(task_credit)
pred

# Evaluate some measures of error
pred$score(msr("classif.acc"))
pred$confusion

# This variable shows available error measures
mlr_measures

# Try computing the Brier score loss ... what is wrong?
# Look at the help file for Learners and see how to rectify this
?Learner


# Uncomment and run the following command first if you do not have the ranger package
# install.packages("ranger")

# Redo everything for a random forest model
learner_rf <- lrn("classif.ranger")

learner_rf$train(task_credit)

pred_rf <- learner_rf$predict(task_credit)

pred_rf$score(msr("classif.acc"))
pred_rf$confusion

# Redefinet the task, this time not getting rid of missing data here and
# specifying what constitutes a positive case
credit_task <- TaskClassif$new(id = "BankCredit",
                               backend = credit_data,
                               target = "Status",
                               positive = "bad")

# Let's see what resampling strategies MLR supports

# The final column are the defaults
as.data.table(mlr_resamplings)
# ... whilst this just gives the names
mlr_resamplings

# To see help on any of them, prefix the key name with mlr_resamplings_
?mlr_resamplings_cv

# The rsmp function constructs a resampling strategy, taking the name given
# above and allowing any options listed there to be chosen
cv <- rsmp("cv", folds = 3)
# We then instantiate this resampling scheme on the particular task we're
# working on
cv$instantiate(credit_task)

# You can see from the documentation that you could access individual folds
# training and testing data via:
cv$train_set(1)
cv$test_set(1)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")

# Have a look at what options and hyperparameters the model possesses
lrn_baseline$param_set
lrn_cart$param_set

# Fit models with cross validation
res_baseline <- resample(credit_task, lrn_baseline, cv, store_models = TRUE)
res_cart <- resample(credit_task, lrn_cart, cv, store_models = TRUE)

# Calculate and aggregate performance values
res_baseline$aggregate()
res_cart$aggregate()

# Remember the error measures we can get ... (or use View() in RStudio for
# nicer format)
as.data.table(mlr_measures)
# ... whilst this just gives the names
mlr_measures

# Again, to see help on any of them, prefix the key name with mlr_measures_
?mlr_measures_classif.ce

res_baseline$aggregate(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr")))
res_cart$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.auc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))

# Use the benchmark functionality to do everything at once, ensuring identical
# settings such as task, folds, etc
res <- benchmark(
  benchmark_grid(
    task        = list(credit_task),
    learners    = list(lrn_baseline,
                       lrn_cart),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
res
res$aggregate()

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# Get the trees (2nd model fitted), by asking for second set of resample
# results
trees <- res$resample_result(2)

# Then, let's look at the tree from first CV iteration, for example:
tree1 <- trees$learners[[1]]

# This is a fitted rpart object, so we can look at the model within
tree1_rpart <- tree1$model

# If you look in the rpart package documentation, it tells us how to plot the
# tree that was fitted
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

# Looking at other rounds from CV
plot(res$resample_result(2)$learners[[3]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[3]]$model, use.n = TRUE, cex = 0.8)

# Enable nested cross validation
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(credit_task, lrn_cart_cv, cv, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[3]]$model)

# Try refitting with a chosen complexity parameter for pruning
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.016)

# Then run this in the benchmark with other options
res <- benchmark(benchmark_grid(
  task       = list(credit_task),
  learners    = list(lrn_baseline,
                     lrn_cart,
                     lrn_cart_cp),
  resamplings = list(rsmp("cv", folds = 3))
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# Trying out pipelines
library("mlr3pipelines")

# Pipelines available (or use View() in RStudio for a nicer look) ...
as.data.table(mlr_pipeops)
# ... whilst this just gives the names
mlr_pipeops

# Again, to see help on any of them, prefix the key name with mlr_pipeops_
?mlr_pipeops_encode

# Uncomment and run the following command first if you do not have the xgboost package
# install.packages("xgboost")

# Create a pipeline which encodes and then fits an XGBoost model
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

# Now fit as normal ... we can just add it to our benchmark set
res <- benchmark(benchmark_grid(
  task        = list(credit_task),
  learners    = list(lrn_baseline,
                     lrn_cart,
                     lrn_cart_cp,
                     pl_xgb),
  resamplings = list(rsmp("cv", folds = 3))
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# First create a pipeline of just missing fixes we can later use with models
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Now try with a model that needs no missingness
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

# Now fit as normal ... we can just add it to our benchmark set
res <- benchmark(benchmark_grid(
  task        = list(credit_task),
  learners    = list(lrn_baseline,
                     lrn_cart,
                     lrn_cart_cp,
                     pl_xgb,
                     pl_log_reg),
  resamplings = list(rsmp("cv", folds = 3))
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

library("mlr3verse")

set.seed(212) # set seed for reproducibility

# Load data
data("credit_data", package = "modeldata")

# Define task
credit_task <- TaskClassif$new(id = "BankCredit",
                               backend = credit_data,
                               target = "Status",
                               positive = "bad")

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(credit_task)

# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop") # This passes through the original features adjusted for
                # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()

# Finally fit the base learners and super learner and evaluate
res_spr <- resample(credit_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

