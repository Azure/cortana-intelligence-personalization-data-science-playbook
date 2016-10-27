# Personalized Offers from Classifiers Use Case

## Executive Summary

To help their customers navigate large websites, many online retailers highlight their offerings by embedding suggestions in each webpage. Users strongly prefer that these advertised offers be tailored to their own interests: useless information creates clutter and detracts from their experience. Retailers can leverage user behaviors like browsing activity, product ratings, and purchases to predict which offers a given user will find appealing. When the set of offers under consideration is relatively small and consistent over time, dense information on user responses to each offer accumulates. This input type is ideal for training classifiers, a machine learning model type with potential advantages including fast response time, lightweight representation, and explainable results. This document introduces data scientists and other indsutry professionals to best practices for offer personalization using classifiers. In addition to providing general advice, this document follows the use case from the [Cortana Intelligence Industry Solution](https://github.com/Azure/Cortana-Intelligence-Suite-Industry-Solutions) on [Personalized Offers in Retail](https://github.com/Azure/Cortana-Intelligence-Suite-Industry-Solutions/tree/master/Marketing/Personalized%20Offers) as an illustrative example.

## Background on Example Retailer: Contoso Mart

Contoso Mart is a fictitious online retailer that has approved a selection of 25 offers for all users. After experimenting with methods to highlight these offers, Contoso Mart has found that users prefer to see only one offer displayed on a web page at a time. Contoso Mart hopes to improve the offer clickthrough rate by displaying the offer deemed most appealing for each user based on the user's recent and longterm browsing history. Since the offers are relatively static and few in number, Contoso Mart's records on user interactions with each offer are dense and well-suited for training a multiclass classifier. Each stage of Contoso Mart's personalized offer development experience will be highlighted in the appropriate section of this document.

## Outline
- [Data Acquisition](#Data-Acquisition)
   - [User Behaviors](#User-Behaviors)
   - [User Descriptors](#User-Descriptors)
   - [Example: Contoso Mart](#dacm)
- [Feature Extraction and Selection](#Feature-Extraction-and-Selection)
   - [Feature Extraction](#Feature-Extraction)
   - [Feature Selection](#Feature-Selection)
   - [Example: Contoso Mart](#fescm)
- [Multiclass Classifier Model Selection](#Multiclass-Classifier-Model-Selection)
   - [Model Types](#Model-Types)
   - [Construction from Binary Classifiers](#Construction-From-Binary-Classifiers)
   - [Implementation](#Implementation)
   - [Example: Contoso Mart](#mscm)
- [Best Practices for Training and Evaluation](#Best-Practices-for-Training-and-Evaluation)
   - [Dataset Partitioning](#Dataset-Partitioning)
   - [Hyperparameter Selection](#Hyperparameter-Selection)
   - [Evaluation Metrics for Multiclass Classifiers](#Evaluation-Metrics-for-Multiclass-Classifiers)
   - [Example: Contoso Mart](#tecm)
- [Operationalization and Deployment](#od)
   - [Creating and Consuming a Web Service](#Creating-and-Consuming-a-Web-Service)
   - [A/B and Multiworld Testing](#ab)
   - [Model Retraining](#Model-Retraining)
   - [Example: Contoso Mart](#odcm)


## Data Acquisition

The first stage in implementation of a classifier for personalized offer recommendations is the collection of data that will be used to train the model. In particular, personalization requires collection of user-specific information such as interests, demographics, and behaviors. The data collected must be sufficient to construct labels (the property of each data point that the classifier will predict) as well as informative features that can be used to predict the labels.

### User Behaviors

**Offer Clickthroughs**

User responses to presented offers -- i.e., whether each offer was clicked or ignored -- constitute the most direct form of evidence for a user's interest or disinterest in an offer. In the most common data collection scheme, a retailer records each offer displayed to a user and each time an offer is clicked: these records can be compared later, using timestamps or unique URIs, to determine which offers were ignored. For technical simplicity, some retailers choose to record only clickthrough events.

Note that offer clickthrough data can only be collected after offers start being displayed on the retailer's website. A retailer can begin collecting data by displaying offers at random or in a targeted fashion. If the distribution of offers displayed to each user is non-random, downsampling (or upsampling) may be used to balance the dataset.

**Page Views**

A user's browsing history may shed light on longterm preferences and current purchase intentions. Each visit to a given page adds evidence of a user's interest in the page's contents, which may correspond to interest in a related offer. A rolling tally of a user's views for each page can be maintained over any time frame of interest, with current values recorded at the time a clickthrough event is logged. Alternatively, each page view can be recorded directly in a log entry containing a timestamp and identifiers for the user and page, and rolling counts can be constructed from these logs.

**Purchases, Product Reviews, and Wishlists**

Interest in some types of offers -- in particular, product recommendations -- correlates with user interest in specific products. Records of users' purchases, reviews, and "wishlist" contents can be used to predict their interest in a product of interest. Raw data of this type is likely to be sparse: for example, the probability that a user will purchase/review/etc. any given product is low. However, dense features appropriate for training classifier models can be constructed by summing across all products in defined categories.

### User Descriptors

Many online retailers request and store customer details that are likely to correlate with their interest in specific offers. For example, customers usually specify their location and gender-specific appellation when entering their shipping information during sign-up or check-out. Some retailers collect additional details like age and stated interests on an opt-in basis, encouraging broad participation by offering incentives. (Users may also volunteer this information once they are convinced that relevant offers will improve their browsing experience.) Additional information may be shared when users link social media accounts containing a user profile.

### Data Acquisition Example: Contoso Mart <a name="dacm"></a>

Contoso Mart sells twenty-five products. Every time a user requests a web page, an advertisement for one of these products is included on the page. (In other words, the type of offer that Contoso Mart hopes to personalize is a product suggestion.) Before implementing personalized offers, Contoso Mart highlighted a randomly-selected offer and recorded the following information for each clickthrough event:
- The product was highlighted in the offer (encoded using 25 one-hot variables)
- The webpage where the offer was displayed (25 possibilities, one corresponding to each product)
- The count of the user's visits to each of the 25 webpages in the past minute, hour, or day (tallies maintained in near-real time using [Azure Event Hub](https://azure.microsoft.com/en-us/documentation/articles/event-hubs-overview/) and [Azure Stream Analytics](https://azure.microsoft.com/en-us/services/stream-analytics/))

The resulting dataset has 101 features. If the number of products/web pages were larger, it might have been advisable to bin products into groups or switch to a hybrid recommender approach.

## Feature Extraction and Selection

The core component of the model training/evaluation datasets is the offer clickthrough data, which generally contains the label of interest (viz., the identifier for the ad that was clicked), the identifier for the user who clicked it, and a timestamp. The user identifier and timestamp can be used to join the dataset with semi-static user descriptions like demographic properties as well as the user's recent behavior at the time the clickthrough occurred. This process may produce a dataset that includes many features, some of which are uninformative or contribute to overfitting. Feature selection can be used to reduce the feature set while maintaining as much predictive power as possible.

### Feature Extraction

**Rolling windows**

Raw event logs typically include information on only one event per row. Rolling window techniques can be applied to these logs to create new features that count the number of events which occurred in the *n* minutes prior to each timepoint. We can then outer join these logs with the main dataset by timestamp (and, if necessary, forward-fill) to annotate the clickthrough data with the relevant rolling event counts. This technique can be used to create potentially-useful features such as "number of user's visits to page x in last hour".

Data scientists who introduce rolling counts during feature extraction must give careful thought to how these features will be computed after the model is deployed. The calculations which they perform to construct features during offline model development may not be ideal in the time-sensitive context of the operationalized model.  Data engineers and solution architects can help design a solution for near-real time calculation of these rolling counts, so that they can be provided directly as inputs to the model. If a processing delay is expected for rolling count calculation, that delay should be simulated during offline feature creation. (Once the solution is operational, rolling counts can be stored with other details of each offer clickthrough, and will no longer need to be generated offline through feature extraction.)

**Inferred User Descriptors**

User descriptors are most useful when they have low "missingness" (fraction of data points with unknown value). Most online retailers collect user interest and demographics on an opt-in basis, resulting in high missingness. The utility of these features can be improved by filling in missing information with an educated guess, e.g. through imputation or classification. For example, information that is commonly provided for shipping purposes, like name and location, could be used to infer other properties like gender, socioeconomic status, or age.

### Feature Selection

During the feature extraction stage, a large number of features may be created from the available data. For example, when employing page view data, we may create a separate feature for rolling counts of page views for each page, in each of multiple time windows. Some of these features will not be correlated to the label of interest (the identifier of the clicked offer) or even detract from a model's predictive power through overfitting. Such features should be removed during the feature selection stage to reduce training time and the potential for model overfitting. The features to retain can be selected using correlation or mutual information with the label, forward selection or backward elimination, and a variety of model-specific approaches (e.g. feature importance for decision forests).

### Example: Contoso Mart <a name="fescm"></a>

Contoso Mart merges all features of interest into the offer clickthrough data during the logging step, ensuring that the model is trained only on features that will be readily available after deployment. Rolling counts of recent page views are maintained using Azure Stream Analytics and Azure Event Hub. Contoso Mart did not choose to include static user attributes (demographics, etc.) in their model, but if a table of user attributes had been available, it could simply be joined to the main dataset by the user identifier field.

## Multiclass Classifier Model Selection

Binary classifiers assign one of two possible labels to a given data point based on the values of relevant features. Multiclass classifiers extend this concept, creating a model that assigns one of 3+ labels for each data point. In this use case, we use a multiclass classifier to assign a label indicating which of the possible offers should be displayed (because that offer is deemed most likely to result in a clickthrough event). 

Major advantages of classifiers over alternatives like hybrid recommendation models include their potentially faster speed, lower resource requirements, and improved explainability. However, classifiers are challenged by the introduction of new classes and very large numbers of classes. (Hybrid recommendation models may be preferable in these cases: se the following example use case.)

### Model Types

Common types of classifier models include:
- [k-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Kernel Machines (e.g. Support Vector Machines, Bayes Point Machines)](https://en.wikipedia.org/wiki/Kernel_method)
- [Random (Decision) Forest](https://en.wikipedia.org/wiki/Random_forest)
- [Decision Jungle](https://www.microsoft.com/en-us/research/publication/decision-jungles-compact-and-rich-models-for-classification/)
- [Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

A number of characteristics should be considered when selecting a classification model:
- Accuracy of the predictions
- Training and scoring speed
- Training and scoring resource requirements
- Availability of confidence metrics for predictions
- Availability of methods for assessing feature importance
- Avalability of model-specific methods to reduce overfitting
- Ability to succinctly explain results

For a detailed comparison of several classifier models, please see our [broader discussion of algorithm selection](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-choice/) or Martin Thoma's blog post on [Comparing Classifiers](https://martin-thoma.com/comparing-classifiers/).

### Construction from Binary Classifiers

Multiclass classifiers can be constructed from sets of binary classifiers in two common ways:

**One vs. One**

Under this scheme, an $n$-class classifier is constructed from $\binom{n}{2} = n(n-1)/2$ binary classifiers. Each binary classifier $f_{ij}: X \to \{ i, j \}$ is trained using the subset of training data points with label $i$ or $j$. The label assigned by the multiclass classifier is the most common label assigned by the set of binary classifiers. Disadvantages of this scheme include the fast growth in number of required classifiers with $n$, and the potential ambiguity when the same number of classifiers support 2 or more most common labels.

**One vs. All**

An $n$-class classifier can also be constructed from $n$ binary classifiers, provided that the classifier returns a score indicating confidence (in addition to an assigned label) for each data point. Each classifier $f_{\ell}: X \to \mathbb{R}$ is trained to return a value indicating its confidence that a data point $\mathbb{x}$ has label $\ell$. Data points are assigned the label corresponding to the most confident classifier, i.e. $f(\mathbf{x}) = \arg\max_{\ell} f_{\ell}(\mathbf{x})$. A disadvantage of this approach is that it requires a classifier type that can assure similarly-scaled confidence scores and increase the probability that classifiers will be trained on imbalanced datasets.

Some binary classifier models have also been extended (with model-specific algorithms) to allow efficient training and scoring without explicitly constructing binary classifiers as described above.

### Implementation

[Azure Machine Learning (AML) Studio](https://studio.azureml.net/) is a cloud-based graphical environment for machine learning data preparation and model development. Many of the multiclass classifier models mentioned above can be incorporated into AML through a code-free drag-and-drop interface, but data scientists can also import R and Python code to implement custom models if they prefer (and share these examples within the community). During model development, intermediate results can be examined using automated summaries, or custom code and visualizations in Python/R Jupyter notebooks. AML also facilitates deployment of predictive web services from trained models.

[Azure App Services](https://azure.microsoft.com/en-us/documentation/services/app-service/) is another option for the deployment of models trained locally. Data scientists can create predictive web services using a wide variety of programming languages and common tools like Flask, Django, and Bottle. The Web App Service can also be used for the construction of web pages that make use of the web service.

### Example: Contoso Mart <a name="mscm"></a>

Contoso Mart chose to implement their classifier model using Azure Machine Learning Studio. A shared experiment showing their model training, scoring, and evaluation process can be found in the [Cortana Intelligence Gallery](https://gallery.cortanaintelligence.com/).

Contoso Mart selected a multiclass logistic regression model based on the following desired features:
- Fast training and scoring
- Low resource requirements
- Availability of $\ell_1$ and $\ell_2$ regularization
- Explainable results

This type of multiclass classifier is one of many available as a built-in module in AML. If Contoso Mart had preferred, they could have created a multiclass model from any available binary classifier module using the [One-vs-All Multiclass module](https://msdn.microsoft.com/en-us/library/azure/dn905887.aspx), selecting a user-contributed [custom module](https://gallery.cortanaintelligence.com/customModules) available in the Cortana Intelligence Gallery, or scripting their own using R or Python.

## Best Practices for Model Training and Evaluation

### Dataset Partitioning

It is common practice to partition the available data points into two sets: a *training set* used to select hyperparameters (if applicable) and train the model, and a *test set* for evaluating the trained model's accuracy. This practice improves the odds that a model's performance on the test set will accurately reflect its performance on future data.

Random splitting of observations into training and test sets may not be ideal for model evaluation. When data points collected from the same user appear in both the training and test set, the model may learn their specific behaviors and thus give more accurate predictions than it would provide for them than for new users. To ensure that the model generalizes well, it may be wise to partition data points at the user level.

Another common partitioning method is to divide observations chronologically: observations collected before a certain date would  form the training set, and all more recent observations would form the test set. This arrangement mimics the real-world scenario in which the model is trained on all data available on a certain date, then tested on new data as it arrives. The performance of the trained model on the test set is realistic in the sense that the model will have no knowledge of any trends that arise after the trainign date. (By contrast, if the data were partitioned randomly, a model would likely be trained using some data points from every time period, and therefore could "learn" all such trends.)

### Hyperparameter Selection

Some classification models take hyperparameters that tune properties such as strength of regularization, learning rate, and number of training rounds. (For more information on which hyperparameters are available for a given classification model, see the documentation for the machine learning model implementation. Descriptions of classification models in AML Studio can be found [here](https://msdn.microsoft.com/en-us/library/dn905808.aspx).) A range of possible hyperparameter values can be tested to identify the values that produce models with optimal performance on a withheld portion of the training set (usually called a "validation" set), which may be statically-defined or varied via [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). In AML Studio, the [Tune Model Hyperparameters](https://msdn.microsoft.com/en-us/library/azure/dn905810.aspx) module can be used to automate hyperparameter selection using cross-validation.

### Evaluation Metrics for Multiclass Classifiers

After the test set has been scored using the trained model, the predicted and actual data point labels can be compared using a variety of metrics:

**Overall and average accuracy**

The *overall accuracy* is the fraction of data points where the predicted and actual labels match. For a test dataset $X$ with predicted labels $Y$, let $X_{\ell}$ denote the set of points with true label $\ell$, and $F_{\ell}$ denote the set of points with predicted label $\ell$. We can then express the overall accuracy using a sum over the $n$ labels:

$$ \mathcal{A}_{\textrm{overall}} = \frac{\sum_{\ell = 1}^n |X_{\ell} \cap Y_{\ell}|}{\left|X\right|} $$

where $|X|$ denotes the size of set $X$. 

An alternative metric is defined in terms of the class-specific accuracies, i.e., the fraction of all data points that are correctly labeled as belonging or not belonging to each class:

$$ \mathcal{A}_{\ell} = \frac{\left|X_{\ell} \cap Y_{\ell}\right| + \left| \, (X \setminus X_{\ell}) \cap (Y \setminus Y_{\ell}) \, \right|}{\left|X\right|}, \hspace{0.5 cm} \ell = 1, 2, \ldots, n$$

The *average accuracy* is the unweighted average of the class-specific accuracies:

$$ \mathcal{A}_{\textrm{average}} = \frac{1}{n} \sum_{\ell=1}^n \mathcal{A}_{\ell}$$

When predictions on a minority class are substantially worse than other predictions -- which may be caused by the small number of training points available -- the average accuracy will be more deeply impacted than the overall accuracy.

**Confusion matrix**

The confusion matrix $\mathcal{C}$ summaries the number of data points in the test set with each combination of true and predicted labels. Each element $c_{\ell m}$ of the confusion matrix is number of data points with true label $\ell$ that were predicted to have label $m$.

As an example, for the following simple dataset and labels:

| True label  | Predicted Label |
|---|---|
| 1 | 2 |
| 1 | 1 |
| 2 | 3 |
| 3 | 3 |
| 1 | 1 |

The corresponding confusion matrix would be:

|   | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 2 | 1 | 0 |
| 2 | 0 | 0 | 1 |
| 3 | 0 | 0 | 1 |

A perfect predictor would produce a diagonal confusion matrix; any non-zero off-diagonal elements correspond to prediction errors. Confusion matrices can be visually inspected or summarized further using metrics like precision and recall.

**Recall**

The model's recall for a class $\ell$ is the fraction of data points in $X_{\ell}$ that were predicted to be in class $\ell$:

$$ \mathcal{R}_{\ell} = \frac{|X_{\ell} \cap Y_{\ell}|}{|X_{\ell}|} $$

There are two common methods for summarizing recall across all classes. The *macro-averaged* recall is the (unweighted) average class recall:

$$ \mathcal{R}_{\textrm{macro-avg}} = \frac{1}{n} \sum_{\ell=1}^{n} \mathcal{R}_{\ell} $$

The *micro-averaged recall*, by contrast, is the a weighted average of the class recalls (with fraction of all datapoints in each class used as the weighting):

$$ \mathcal{R}_{\textrm{micro-avg}} = \sum_{\ell=1}^{n} \frac{|X_{\ell}| \mathcal{R}_{\ell}}{|X|} $$

The micro- and macro-averaged recall are identical for balanced datasets. In imbalanced datasets, poor recall on a minority class has a more dramatic effect on macro- than micro-averaged recall.

**Precision**

The model's precision for a class $\ell$ is the fraction of data points predicted to be in class $\ell$ that are truly in class $\ell$:

$$ \mathcal{P}_{\ell} = \frac{|X_{\ell} \cap Y_{\ell}|}{|Y_{\ell}|} $$

As with recall, precision for multiclass classifiers can be calculated using micro- or macro-averaging:

$$ \mathcal{P}_{\textrm{macro-avg}} = \frac{1}{n} \sum_{\ell=1}^{n} \mathcal{P}_{\ell}, \hspace{1 cm}  \mathcal{P}_{\textrm{micro-avg}} = \sum_{\ell=1}^{n} \frac{|Y_{\ell}| \mathcal{P}_{\ell}}{|Y|} $$

For additional description of these and other metrics, see [Computing Classification Evaluation Metrics in R](http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html) by Said Bleik.

### Example: Contoso Mart <a name="tecm"></a>

Contoso Mart uses Azure Machine Learning Studio to train and evaluate its classifier. The metrics and confusion matrix described above are automatically generated by the Evaluate Model module, and can be inspected by right-clicking on the module's output port to select the visualization option.


If desired, Contoso Mart could also calculate arbitrary metrics of interest using Python and R scripts they create or find online.

For additional information on model evaluation in AML Studio, please see Gary Ericson's [How to evaluate model performance in Azure Machine Learning](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-evaluate-model-performance/)

## Operationalization and Deployment <a name="od"></a>

### Creating and Consuming a Web Service

After evaluation reveals that a trained model is fit for operationalization, a web service can be created to surface its recommendations. Depending on how the web service is deployed, it may be necessary to explicitly define the web service's input and output schema, i.e., the input data format needed to call the web service, and the format of the results to be returned. In Azure Machine Learning Studio, a predictive web service can be created from a trained model with a single click and tested using a graphical interface, Excel plug-in, or automatically-generated sample code. Developers using Azure Web Apps can also use common web service creation tools like [Flask](https://azure.microsoft.com/en-us/documentation/articles/web-sites-python-create-deploy-flask-app/), [Bottle](https://azure.microsoft.com/en-us/documentation/articles/web-sites-python-create-deploy-bottle-app/), and [Django](https://azure.microsoft.com/en-us/documentation/articles/web-sites-python-create-deploy-django-app/).

Many retailer websites can be modified to request a personalized offer recommendation from the web service during page loading; the recommended offer's image and link can then be incorporated into the webpage in time for rendering.

### A/B and Multiworld Testing  <a name="ab"></a>

Since personalized offers often contribute significantly to revenue and are displayed prominently, online retailers may wish to gradually introduce a new recommendation model to ensure that performance and quality meet expectations. In the A/B testing scheme, the new model's suggestions can be served to a small subset of users while the previous model continues to be displayed to the majority. Offers provided to the two user groups can then be compared on metrics like loading speed, error rate, and clickthrough rate to ensure that the new model is equally reliable and offers superior product suggestions.

[Multiworld Testing](https://www.microsoft.com/en-us/research/project/multi-world-testing-mwt/) is an extension of A/B testing that permits multiple models to be tested concurrently, with automated selection of effective models. The [Multiworld Decision Service](http://mwtds.azurewebsites.net/) simplifies the process of incorporating this form of testing into a web application.

### Model Retraining

The multiclass classifier should be retrained as the selection of available offers changes and additional training data accumulate. This retraining step may be supervised manually by a data scientist or performed programmatically. If desired, the retrained model can be compared directly to the previous model using A/B testing before it is fully deployed. More information on programmatic retraining in Azure Machine Learning Studio is available here: [Retrain Machine Learning models programmatically](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-retrain-models-programmatically/).

### Example: Contoso Mart  <a name="odcm"></a>

Each time a user requests to load a web page, Contoso Mart's web app calls the predicted AML web service to request a recommendation. The web service returns the identifier for a personalized offer, which is used during page rendering to add the offer's image and hyperlink to a page template. To monitor the personalized offers' efficacy, Contoso Mart continues to record all clickthrough events to Azure Blob Storage, from which they are regularly transferred to SQL Data Warehouse for long-term storage, analysis, and display via a Power BI dashboard.

![Image of Contoso Mart's Solution Architecture](https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/img/architecturediagram.jpg)
