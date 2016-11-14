# Product Suggestions Using Hybrid Recommenders

## Executive Summary

Targeted product suggestions based on prior behavior and demographics can help users identify items of interest in large catalogs. In addition to driving increased revenue, such suggestions improve the user's experience by replacing distracting ads with relevant information: in a [recent survey](https://www.listrak.com/about/news-events/Press-Release-Survey-Reveals-80-Percent-of-Email-Readers-Find-It-Useful-When-Emails-Feature-Recommen/), 67% of shoppers enjoyed seeing product recommendations on a retailer's website while shopping. Hybrid recommenders combine collaborative filtering with background information on customers and products to generate high-quality product recommendations on large, sparse datasets while gracefully handling the addition of new products and users over time.

## Sample Use Case: Contoso Mart

Contoso Mart is a fictitious online retailer with a catalog of ~100k products and ~200k users. To help their customers identify relevant products in its large catalog, Contoso Mart plans to embed product suggestions -- clickable thumbnails and descriptions of products a user may enjoy -- into each page of its website. Contoso Mart measures success on its product suggestions with two metrics: clickthrough rate (the fraction of product suggestions that are clicked) and rate of conversion (the fraction of clickthroughs that ultimately lead to a product purchase). Under the current product suggestion system, new users are often frustrated with the poor quality of recommendations they receive, and new products rarely appear in recommendations. To improve suggestions for new users and products, Contoso Mart has decided to augment its existing collaborative filtering solution with content-based filtering.

## Outline
- [Data Acquisition](#dataacquisition)
   - [Evidence of User-Product Affinity](#evidence)
   - [User Descriptions](#userdescriptions)
   - [Product Descriptions](#productdescriptions)
   - [Product Recommendation Quality Assessment](#quality)
- [Feature Extraction](#featureextraction)
   - [Affinity Scores](#affinityscores)
   - [User Description Augmentation](#augmentation)
- [Hybrid Recommender Model Selection](#modelselection)
   - [Matchbox Recommender in Azure Machine Learning Studio](#matchbox)
   - [Recommendations API](#api)
   - [Custom Implementation](#custom)
- [Best Practices for Model Training and Evaluation](#bestpractices)
   - [Evaluation Set Creation](#evaluationset)
   - [Hyperparameter Selection](#hyperparameters)
   - [Evaluation Metrics](#evaluation)
- [Deployment](#deployment)
   - [Creating and Consuming a Web Service](#webservice)
   - [A/B and Multiworld Testing](#ab)
   - [Model Retraining](#retraining)

<a name="dataacquisition"></a>
## Data Acquisition

<a name="evidence"></a>
### Evidence of User-Product Affinity

The core input data for collaborative filtering algorithms are estimates of the affinity between users and products. Many user actions on online retail websites can provide evidence of affinity to a product, including:
* Purchasing the product
* Leaving a favorable review
* Adding the product to a "wishlist"
* Clicking an ad for the product
* Viewing a product's description page

Similarly, some actions suggest an absence of affinity:
* Leaving a negative review
* Returning the product after a purchase
* Ignoring an ad for the product

Below, we list specific considerations for collecting each type of data mentioned above. A few general considerations apply to all data types:
- User preferences may shift over time. If desired, older observations can be assigned a reduced weight during affinity score calculation.
- Retailers with large product catalogs will likely find that the average user interacts with a small fraction of products. It is unnecessary (and resource-intensive) to store null observations.

**Sales Transactions**

Purchases are strong indicators of affinity between users and products. Purchase information may need to be mined from sales transaction data, which may contain sensitive (and for this purpose, extraneous) information on payment method or shipping address. At Contoso Mart, sales transactions are stored in semi-structured JSON form to accommodate the possibility of multiple products per transaction, e.g.:

```
{
    "TransactionID": 123456789,
    "UserID": 89643,
    "TransactionDateTime": 10/11/2016 14:32:12,
    "Products": [
        {
            "SKUID": "981745BLACKGOLD",
            "ProductID": 981745,
            "ProductName": "Acme RC Monster Truck",
            "OriginalPrice": 19.99,
            "Discount": {}
        },
        {
            "SKUID": "398676EXPIRY20201231",
            "ProductID": 398676,
            "ProductName": "Conglomo AA Batteries (12 ct)",
            "OriginalPrice": 9.99,
            "Discount": {
                "DiscountName": "10% off sale",
                "DiscountType": "PercentageReduction",
                "DiscountAmt": 1.00
            }
        }
    ],
    "SubTotal": 28.98,
    "Tax": 2.03,
    "Total": 31.01,
    "PaymentMethods": [
       ...
    ]
}
```

Semi-structured data may be flattened into relational form, or processed in its native form using appropriate sofware: [SQL syntax and several programming languages](https://azure.microsoft.com/en-us/documentation/articles/documentdb-introduction/) can be used to interact with DocumentDB NoSQL databases, and most programming languages offer packages for parsing raw JSON files. For retailers that track items using both stock-keeping unit (SKU) and product identifiers, we recommend providing suggestions at the product level. Pairs of user and product identifiers should be extracted from the sales transaction data, along with the date of purchase if desired.

**Reviews and Product Returns**

Many retailers allow users to submit product reviews regardless of whether the user purchased each product through the retailer's own website. This permits users to express many strong preferences/aversions that can be leveraged to improve the quality of product recommendations. User and product identifiers, review date, and the numeric/ordinal rating are typically the most informative fields, but [sentiment analysis and other text analytics](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api) may be applied to the free-form text portion of the review if desired. Similar data may be collected for product returns.

**Wishlist Entries**

Some retailers offer a "wishlist" feature where users can record products they aspire to buy in the future. Recorded product suggestions can be harnessed to remind users of their intentions, increasing the probability of conversion to purchase. Each wish's user and product identifiers, as well as the date of addition to the wishlist, can be extracted from these records.

**Countable Events: Ad Clickthroughs and Product Description Page Visits**

A user responds to a product recommendation/advertisement either by clicking the provided link (which normally leads to the product's description page) or ignoring it. Both outcomes provide information about the user's level of interest in the product, and additional evidence accumulates each time the same recommendation is displayed: tallies of "click" vs. "no-click" events can be maintained or calculated post hoc from logs.

Users may also visit a product's description page after following an external link or clicking on a search result. Particularly interested customers may visit the same description page multiple times: the visit count is therefore informative of a user's affinity for the product. 

<a name="userdescriptions"></a>
### User Descriptions

New users at Contoso Mart may not yet have made purchases, left ratings, or navigated the website. Personalized product recommendations can nonetheless be provided using an approach called *content boosting*, which leverages the new user's demographic similarity to existing users of the website. To employ content boosting, information that can be used to assess the similarity between users must be supplied.

Some user descriptors are straightforward to infer from data collected during account creation. For example, a user's shipping address suggests their location, and their chosen appellation (Mr./Ms./Mrs.) reflects their gender. Though age, marital status, and interests are highly relevant for product recommendations, most retailers do not request this information during account creation because it may be considered overly personal or extend the sign-up process unnecessarily. Instead, some retailers request this information after sign-up with the explanation that it will be used to provide more accurate product recommendations. Additional incentivization can be provided by offering a Rewards program that requires users to share additional demographic data.

Temporary or redundant accounts typically include only a fraction of a user's purchases and other behaviors. When multiple accounts can be attributed with high confidence to a single customer, recommendation quality may be improved by combining activities across accounts. Information available in user descriptions, including full name and address, may be used to identify candidate merges. Infrequently-used accounts that cannot be merged are prime candidates for removal if subsampling is required for speed/resource management during training.

<a name="productdescriptions"></a>
### Product Descriptions

Like new users, new products pose a challenge for the recommender because no purchase or rating data will be available for these items. Content boosting can also be performed if information is provided for assessing similarity between products. Generic product descriptors include:
* Product ontology: many retailers group products into departments (Home & Garden, Electronics, ...) and further into categories and subcategories.
* Brand or manufacturer
* Price
* Weight (with price-to-weight ratio used to normalize value for products sold in multiple quantities)

<a name="quality"></a>
### Product Recommendation Quality Assessment

After recommendations have been incorporated into the website, retailers should begin to collect data on their efficacy. Contoso Mart assesses the quality of its product recommendation system using two metrics:
* Clickthrough rate: the fraction of product recommendations that are clicked
* Conversion rate: the fraction of product recommendations that result in purchases

To calculate these metrics, one must know which recommendations were displayed to each user, whether or not each recommendation was clicked, and whether or not each recommendation resulted in a purchase (within, say, 24 hours of the recommendation being displayed). In Contoso Mart's case, clickthrough data and sales transaction data are stored separately and must be joined together.

<a name="featureextraction"></a>
## Feature Extraction and Selection

<a name="affinityscores"></a>
### Affinity Scores

The standard input for collaborative filtering algorithms is a list of user-product pairs along with a calculated affinity score:
```
UserID  ProductID  Affinity
   1        1         10
   2        3          3
   3        2          7
  ...      ...        ...
```
To arrive at an affinity score, retailers typically combine multiple forms of evidence -- including purchases, ratings, returns, wishlists, recommendation clickthroughs, and product description page visits -- using a combination of analytics and industry knowledge. Contoso Mart calculates affinity scores as a weighted sum of purchase count, product recommendation clickthrough count, and number of ignored product recommendations. The weightings for each event type in affinity score calculation can be explored using a hyperparameter search as described in the [Training and Evaluation](#Training-and-Evaluation) section below.

<a name="augmentation"></a>
### User Description Augmentation

Users living in the same geographic region are more likely to share weather conditions, culture, age, and socioeconomic status than users living far apart. Location may be a highly informative feature when identifying similarities between users. Assessing proximity between users may require [geocoding](https://msdn.microsoft.com/en-us/library/ff701713.aspx), the extraction of geographical coordinates from e.g. shipping addresses. The raw coordinates (or categories labeling arbitrary geographical regions) can be provided as features for content-based filtering.

Some retailers produce classifiers that partition users into groups that may share product affinities. For example, a classifier may attempt to predict a user's sex from their first name or identify current/expectant parents. These classifiers can replace intrusive solicitation of users' personal information to generate features for content-based filtering.

<a name="modelselection"></a>
## Hybrid Recommender Model Selection

Hybrid recommenders produce suggestions based on two forms of filtering:
- *Collaborative filtering* employs behavioral (purchase, rating, etc.) data to map users and products to a low-dimensional space where their proximity can be quickly assessed. It is found to perform well with the big, sparse datasets expected for retailers with very large numbers of users and products.
- *Content-based filtering* enables personalized recommendations for brand-new users with no accumulated behavioral data by wielding user demographic data (and other descriptions) to identify similar users. This approach also permits recommendations for products that have not yet been sold.

The Azure ecosystem provides several options for implementing hybrid recommenders in Azure: code-free web service deployment in Azure Machine Learning Studio, training and deployment using the Recommendations API, and hosting a custom implementation of a hybrid recommender model.

<a name="matchbox"></a>
### Matchbox Recommender in Azure Machine Learning Studio

[Azure Machine Learning (AML) Studio](https://studio.azureml.net/) is a graphical environment for analytics and web service production. AML Studio's built-in hyrid model, the [Matchbox Recommender](https://www.microsoft.com/en-us/research/publication/matchbox-large-scale-bayesian-recommendations/), can be trained and stored, used to create a web service that quickly generates product suggestions for a user of interest during page loading, and programmatically retrained on a scheduled basis. The Matchbox Recommender's major advantages over other methods discussed below include code-free implementation and avoidance of recommendations matching prior observations; as of this writing, however, the Matchbox Recommender is limited to 10 GB of training data.

<p align="center">
<img src="https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/img/hybrid_recommender/screenshots/hybrid_recommender_graph.PNG?raw=true"></p>

The Matchbox Recommender can be used to predict affinties of specific user-product pairs (useful during model validation/evaluation) or to supply a specified number of item recommendations for a specific user (for operationalization). The responsibility of combining all known behaviors for each user-product pair into a single affinity score lies with the retailer. AML Studio provides a wide variety of intuitive tools for model validation and web service deployment.

Examples of recommender systems built with the Matchbox Recommender can be examined and deployed from the [Cortana Intelligence Gallery](https://gallery.cortanaintelligence.com) with a free account:
- [Product Recommendations via Hybrid Recommender](http://gallery.cortanaintelligence.com/Experiment/Product-Recommendations-via-Hybrid-Recommender-1) 
- [Personalized restaurant suggestions based on customer reviews](https://gallery.cortanaintelligence.com/Tutorial/8-Recommendation-System-1)
- [Movie recommendations based on viewer ratings](https://gallery.cortanaintelligence.com/Experiment/Recommender-Movie-recommendation-3)

<a name="api"></a>
### Recommendations API

The [Recommendations API](https://www.microsoft.com/cognitive-services/en-us/recommendations-api) is a [Microsoft Cognitive Service](https://www.microsoft.com/cognitive-services/en-us/apis) implementing a hybrid recommender model. Programs written in Python, Java, C#, and many other languages can interact with the API to upload data, train the hybrid recommender model, and quickly receive recommendations for a specific user. Scripts which interact with the Recommendations API can be incorporated directly into [Azure Web Apps](https://azure.microsoft.com/en-us/documentation/articles/app-service-web-overview/).

The Recommendations API offers several features not available in the Matchbox Recommender:
- Unlike the Matchbox Recommender, there is currently no data size limit.
- User behaviors -- such as purchasing, recommending, or clicking a product's ad -- can be reported directly to the recommender via the API. (An affinity score can be automatically computed from these behaviors.)
- Rules specifying items which should never appear in recommendations, and which items should be promoted, can be specified.
- Products which a user has already purchased or viewed may still be recommended to the user.

The following examples illustrate the use of the Recommendations API:
- [A sample recommender implementation in C#](https://code.msdn.microsoft.com/Recommendations-144df403)
- [Product Recommendation in SalesForce using Microsoft Cognitive API](https://gallery.cortanaintelligence.com/Collection/Product-Recommendation-in-SalesForce-using-Microsoft-Cognitive-API-1)

The following case studies describe customers' experiences implementing product suggestions using the Recommendations API:
- [Youbookx (includes comparison to Google Prediction API)](https://customers.microsoft.com/en-US/story/youboox-can-recommend-the-perfect-book-for-you-and-the)
- [MEO](https://customers.microsoft.com/en-US/story/meocustomerstory)
- [AllRecipes.com](http://advocacypublic.clouddam.microsoft.com/en-US/story/top-cooking-website-energizes-a-new-generation-of-cooks-with-personalized-recommendations)
- [JJ Foods Inc.](https://customers.microsoft.com/en-US/story/food-delivery-service-uses-machine-learning-to-revolut)

For more information on the Recommendations API, please see:
- [Recommendations UI](https://recommendations-portal.azurewebsites.net/#/projects)
- [UI Quick Start](https://azure.microsoft.com/en-us/documentation/articles/cognitive-services-recommendations-ui-intro/)
- [API Quick Start](https://azure.microsoft.com/en-us/documentation/articles/cognitive-services-recommendations-quick-start/)
- [API Sample](https://github.com/microsoft/Cognitive-Recommendations-Windows)
- [API Reference](https://westus.dev.cognitive.microsoft.com/docs/services/Recommendations.V4.0/operations/56f30d77eda5650db055a3db)

<a name="custom"></a>
### Custom Implementation

Recommendation systems are under active research in machine learning. While many retailers employ Azure's out-of-the-box offerings, custom machine learning approach could be rapidly operationalized using [Azure Web Apps](https://azure.microsoft.com/en-us/documentation/articles/app-service-web-overview/) or [Azure Machine Learning Studio](https://studio.azureml.net/) if desired. These Azure services support common programming languages like Python, R, Java, and .NET, and allow the integration of open source packages for model training and scoring.

<a name="bestpractices"></a>
## Best Practices for Model Training and Evaluation

<a name="evaluationset"></a>
### Evaluation Set Creation

Accurate evaluation of a trained model's performance requires the use of fresh data points that were not part of the training set. Contoso Mart reserved a portion of the available user-product affinity scores to create a "test set" of data points for use in model evaluation; the remaining affinity scores formed the "training set" used for hyperparameter selection and model training. This standard practice improves the odds that a model's performance on the test set will generalize to future data.

Random splitting of observations into training and test sets may not be ideal for model evaluation. Contoso Mart hoped to be able to compare the performance on the model on brand new products/users vs. products and users who had many observations in the training set ("frequently-returning users"). This was made possible by ensuring that a substantial fraction of users/products were only observed in the test set. Models deployed in Azure Machine Learning Studio may use the Split Data module's Recommender Split mode to automate this form of partitioning.

<p align="center"><img src="https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/img/hybrid_recommender/screenshots/hybrid_recommender_split_data_module.PNG?raw=true"></p>

Another common method for observation partitioning is to divide observations chronologically. Unlike a random split, the chronologically-defined training set would not contain any observations that might unfairly "reveal" future trends during model training. This mimics the real-world scenario in which the model is trained on all data available on a certain date, then tested on new data as it arrives. Despite these potential advantages, Contoso Mart chose not to perform a chronological split because evaluating the model would require recalculating affinity scores as additional evidence (user behaviors like purchases and reviews) accumulated over time.

<a name="hyperparameters"></a>
### Hyperparameter Selection

**Affinity Score Calculation Parameters**

Many hybrid recommender implementations require that an affinity score be supplied for each user-product pair of interest. The formulas retailers use to combine purchase, clickthrough, review, etc. records into a single affinity score are often the result of the retailers' own analytical investigations and may be considered confidential. Contoso Mart uses a formula of the following form:

<p align="center">
<img src="https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/img/hybrid_recommender/eqns/affinity_score.PNG?raw=true"></p>

where *b* is a constant bias term, *c<sub>i</sub>* is the count of times that behavior *i* was performed for the given user-product pair, and *w<sub>i</sub>* are the weights which determine how strongly each behavior type contributes to the affinity score. (Notice that the affinity scores have been rectified to produce scores that always lie in the range 1-10.) Contoso Mart has applied insider knowledge of the industry to make educated guesses about possible bias and weight values. Cross-validation may be used to compare model performance using a small number of possible parameter sets, with the optimal set used to train the final model.

**Latent Dimension and Training Round Count Parameters**

Many collaborative filtering models, including the Matchbox Recommender, fit a representation for each user and product in a low-dimensional space. Both performance and runtime will generally rise with the number of latent dimensions (sometimes referred to as "traits") and training rounds. It may also be necessary to perform training in batches to limit memory requirements.

<p align="center">
<img src="https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/img/hybrid_recommender/screenshots/hybrid_recommender_train_module.PNG?raw=true"></p>

After selecting an affinity score calculation method, one may examine the trade-off between performance and runtime to make an informed selection of parameters. Cross-validation can be applied again to test a small number of possible parameter combinations. The parameter combination which maximizes performance is used to train a model using the complete training data set.

<a name="evaluation"></a>
### Evaluation Metrics

During hyperparameter selection and ultimately model evaluation, it is necessary to define a metric for assessing model performance. The Evaluate Recommender module provides the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between the true affinity scores and those predicted by a trained Matchbox Recommender's "Predict Ratings" feature.

In general, affinity score predictions on the test set will be more accurate for users/products that were present in the training set. Contoso Mart used an Execute Python Script module to compare the RMSE metric for test set observations with never-before-seen users and products to observations where both product and user were present in the training set. While affinity score predictions were less reliable for new user/products, content-based filtering did improve the quality of these predictions. (The performance with and without content-based filtering can be compared by simply disconnecting the user and product descriptions from the Train/Score Matchbox Recommender modules in the experiment graph.)

<a name="deployment"></a>
## Deployment

<a name="webservice"></a>
### Creating and Consuming a Web Service

The first step in operationalizing the hybrid recommender's product suggestions is the creation of a predictive web service. Because Contoso Mart implemented their model in Azure Machine Learning Studio, they were able to generate the corresponding predictive web service with a single click. The web service's functionality can be tested using sample code, a web interface, or an Excel plug-in.

Many retailer websites are created using common programming languages that support calling and parsing responses from web services. The web service's sample code snippet can be integrated into the existing website code to request a product recommendation each time a user loads a webpage, and incorporate that recommendation into the displayed page. To reduce lag and ensure variability in recommendations, some retailers request multiple recommendations for each active user and store these for quick access during future page rendering.

<a name="ab"></a>
### A/B and Multiworld Testing

Product suggestions are vital drivers of online retail revenue: new recommender models can be gradually introduced to minimize any negative impact if their real-world performance proves to be suboptimal. After creating a hybrid recommender, a retailer may begin displaying its product suggestions to a small subset of users while continuing to show the previous model's suggestions to all other users. This common practice, called A/B testing, allows the retailer to compare speed and quality (e.g. clickthrough/conversion rate) metrics of both recommenders while controlling for other factors. When the retailer is convinced that the hybrid recommender model is reliable and offered superior product suggestions, it can begin providing the new model's suggestions to all users.

[Multiworld Testing](https://www.microsoft.com/en-us/research/project/multi-world-testing-mwt/) is an extension of A/B testing that permits multiple models to be tested concurrently, with automated selection of effective models. The [Multiworld Decision Service](http://mwtds.azurewebsites.net/) simplifies the process of incorporating this form of testing into a web application.

<a name="retraining"></a>
### Model Retraining

The hybrid recommender can be retrained programmatically to incorporate recent behavior and new users/products. If desired, the retrained model can be compared directly to the previous model using A/B testing before it is deployed. More information on programmatic retraining is available here: [Retrain Machine Learning models programmatically](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-retrain-models-programmatically/).
