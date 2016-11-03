# Frequently Bought Together Product Suggestions

## Executive Summary

Customers often purchase complementary products in a single order. Forgetting to include relevant accompaniments can leave other items temporarily inoperable or unenjoyable, decreasing satisfaction and necessitating a last-minute purchase of the missing items. Fortunately, relevant products can be suggested to customers based on their other intended purchases using co-occurence patterns learned from sales transaction data. These "frequently bought together" product suggestions drive both revenue and customer satisfaction.

Unlike product recommendations based on long-term user characteristics like past purchases and demographics, "frequently bought together" suggestions are personalized using a user's very recent shopping behavior, viz., which items are currently in their shopping cart. (Historical user-specific features are less informative when predicting forgotten products in the current transaction.) Many online retailers display both types of product suggestions to consumers because they satisfy complementary goals.

## Example: Contoso Mart

Contoso Mart is an online retailer that has noticed a disturbing recent trend in its electronics department. Over the holiday season, many customers found that they could not use toys/gadgets received as gifts because the manufacturer did not include batteries or a charger. While the product descriptions clearly indicated that these items were sold separately, dissatisfaction has driven negative reviews and product returns. Connie, CEO of Contoso Mart, further suspects that sales of these accessory items are being lost to brick-and-mortar retailers as customers choose not to wait for a second shipment.

Connie hopes to recapture this revenue and improve the customer experience by including complementary-product suggestions in the "shopping cart" view of Contoso Mart. The sheer number of products and their rapid turnover prohibit manual curation: automatic detection of complementary products will be necessary.

## Outline
- [Data Acquisition](#Data-Acquisition)
   - [Sales Transaction Data](#Sales-Transaction-Data)
   - [Optional: Product and Transaction Descriptions](#Optional)
- [Model Selection](#Model-Selection)
   - [Item-to-Item Collaborative Filtering](#Item-to-Item-Collaborative-Filtering)
   - [Association Rules](#Association-Rules)
- [Best Practices for Model Training and Evaluation](#Best-Practices-for-Model-Training-and-Evaluation)
   - [Evaluation Set Creation](#Evaluation-Set-Creation)
   - [Hyperparameter Fitting](#Hyperparameter-Fitting)
   - [Evaluation](#Evaluation)
- [Operationalization and Deployment](#Operationalization-and-Deployment)
   - [Creating Predictive Web Services from Trained Models](#Creating-Predictive-Web-Services-from-Trained-Models)
   - [A/B Testing](#ab)
   - [Model Retraining](#Model-Retraining)

## Data Acquisition

The relevant input data for this use case are transaction-level sales records from which we can extract which products have been purchased together. The volume of transaction data needed to accurately detect "frequently bought together" patterns will vary with the distributions of:
* Number of items per transaction: for example, single-item purchases are non-informative
* Number of transactions per product: pattern inference is challenging for new or infrequently-purchased products
* Purchasing pattern strength: patterns may be weaker for products sold in bulk, with long shelf lives, and/or multiple substitutes

When sales transaction data are not available in sufficient quantity, synthetic transactions reflecting expected purchase patterns may replace or augment historical records.

### Sales Transaction Data
Like many online retailers, Contoso Mart stores its transactional data in a semi-structured format to accommodate the variability in number of products per transaction (among other advantages). Typical sales records resemble the following:
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
Like many retailers, Contoso Mart tracks its offerings in a hierarchical scheme of identifiers. The most specific descriptor for an item is its *stock keeping unit* identifier (`SKUID`), used for tracking inventory. The next level of organization, the product ID, is used to group together one or more SKUs that are qualitatively similar and always sold at the same price. For example, the different sizes of a certain t-shirt would share a product ID since customers expect to pay the same price for any size, but each size would have a unique SKUID so that its inventory can be managed independently. Retailers may further group products by product group, brand, or department.

Understanding a retailer's identifier hierarchy is necessary to select a level at which to aggregate data and make recommendations. In Contoso Mart's case, it seems appropriate to make "frequently bought together" selections based on product IDs rather than SKUIDs. This choice offers two major benefits: (i) we will have increased power for accurate pattern detection because the number of observations per product ID is larger, and (ii) customers will receive a suggestion to purchase a product and can select their preferred SKU themselves.

### Optional: Product and Transaction Descriptions <a name="Optional"></a>

As we will describe further in the Model Selection section below, new products pose a challenge to recommender systems because they have no/few training observations on which to learn purchase patterns. Some recommender models can use a supplied list of product details to identify current items that are similar to a new item; this information can be used to produce appropriate recommendations for the new item even before the first sale has occurred. Most retailers maintain a product catalog containing details such as brand, ontology (place within a product hierarchy, e.g. "baked goods, bread, pre-sliced loaves"), seasonality, etc. that can be used for this purpose. Sale transaction details like time of day, day of week, number of products, total price, etc. can also help identify similarities between transactions, potentially improving recommendations.

## Model Selection

### Item-to-Item Collaborative Filtering

Collaborative filtering algorithms can leverage large, sparse datasets to produce recommendations. Most such models expect input in the form of triplets representing a user id, product id, and a "rating": a numerical indication of the affinity between the user and product that may represent an actual product review or some other behavior like a product purchase or return. Collaborative filtering algorithms map each product and user onto a low-dimensional latent feature space where affinity can be estimated by proximity. While collaborative filtering is canonically used to provide product recommendations for a specific user, it can also be used to provide product recommendations based on other products, an approach often called "item-to-item collaborative filtering."

For "frequently bought together" suggestion use case, we can group products into transactions rather than users. Inclusion of a product in a transaction can be reflected by assigning a high rating for the pair. It is also necessary to include some "negative examples" -- product-transaction pairs that were not observed -- to which a low rating is assigned. (The collaborative filtering algorithm will not function if all ratings are identical.) It is not efficient to include all possible negative examples, so a random selection is chosen for inclusion. After parsing the sales transaction data and adding negative examples, the final input for collaborative filtering might resemble the following:

```
transactionid  productid   rating
      1          juice        5
      1         cookies       5
      1         kippers       1
      2          chips        5
      2         spinach       1
```
This sample data could be interpreted as showing that the first transaction actually contained juice and cookies, and kippers are a randomly-selected product included as a negative example.

Hybrid methods that combine collaborative filtering with content-based filtering can be used to create recommenders accommodate new products without requiring retraining. Supplied details of the new product are used to identify similar products with which "frequently-bought-together" recommendations can be made. Many retailers maintain the needed information in their product catalog. Features of individual transactions can also be supplied if desired.

Several options for training and deploying collaborative filtering models are available on Azure:

[Azure Machine Learning Studio](https://studio.azureml.net/), a graphical development environment for analytics and web service design, includes a collaborative filtering model called a **[Matchbox Recommender](https://www.microsoft.com/en-us/research/publication/matchbox-large-scale-bayesian-recommendations/)**. Matchbox Recommenders can provide item-to-item recommendations and may incorporate content-based filtering if desired. As of this writing, the Azure Machine Learning Studio environment is limited to 10 GB of data storage. Examples of recommender systems built with the Matchbox Recommender can be examined and deployed from the [Cortana Intelligence Gallery](https://gallery.cortanaintelligence.com) with a free account:
- [An example of personalized restaurant suggestions based on customer reviews](https://gallery.cortanaintelligence.com/Tutorial/8-Recommendation-System-1)
- [An example of movie recommendations based on viewer ratings](https://gallery.cortanaintelligence.com/Experiment/Recommender-Movie-recommendation-3)

The **[Recommendation API](https://www.microsoft.com/cognitive-services/en-us/recommendations/documentation)** is a [Microsoft Cognitive Service](https://www.microsoft.com/cognitive-services) API that can be used to remotely train and store a collaborative filtering model (with or without content-based filtering). While some coding experience is required to interact with the API, it offers increased flexibility including the ability to operate on larger datasets and to set rules about which products should be actively promoted or not recommended.

Custom code implementing other forms of collaborative filtering or hybrid recommenders may also be trained or operationalized on Azure, e.g. using [Azure App Service](https://azure.microsoft.com/en-us/services/app-service/) or [Azure Machine Learning Studio](https://studio.azureml.net/).

### Association Rules

Association rules describe strong (but not necessarily perfect) patterns such as, "If *x* is purchased in a transaction, then *y* will also be purchased in the same transaction." These rules are often abbreviated with the shorthand *x* -> *y*. Such patterns in sales transaction data can be exploited be recommending *y* once *x* has been added to a shopping cart. *x* and *y* may be individual products, but in general they can represent sets of products, e.g.

\{ graham crackers, marshmallows  \} -> \{ chocolate  \}

An association rule's utility is generally assessed by two metrics:
* The rule's *support* is the fraction of transactions containing both *x* and *y*. If the support is very low, then there cannot be very much evidence to support the rule.
* The rule's *confidence* is the fraction of transactions containing *x* that also contain *y*. Low confidence indicates that the rule is often broken.

Descriptions of additional association rule metrics can be found in the [Association Analysis chapter](http://www-users.cs.umn.edu/~kumar/dmbook/ch6.pdf) of [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php) by Tan, Steinbach, and Kumar.

When making personalized offers, one may be willing to tolerate relatively low confidence values. For example, if 20% of customers who buy ketchup also buy mustard, the rule \{ ketchup \} -> \{ mustard \} will be broken 80% of the time, but the purchasing trend is still strong enough to merit a recommendation. By contrast, a lower bound on support can help avoid a large number of "false positive" association rules that would cause confusion for users.

Downsides of the association rule approach include:
- Long runtimes may be required when the retailer's product catalog is large. (The rate of runtime growth with product number can be reduced by setting a maximum number of items that may be included in a single rule.)
- Association rules should be recreated when new products are added: unlike hybrid recommenders, association rules cannot recommend new products on the basis of similarity to other products in the catalog.

Association rule mining packages are available in R (e.g., [arules](https://cran.r-project.org/web/packages/arules/index.html)) and Python (e.g., [apriori](https://github.com/asaini/Apriori)). These packages typically require that the product identifiers in each transaction be provided in the form of a list: in Contoso Mart's case, this requires parsing each semi-structured sales transaction record to obtain a variable-length list of the products sold in the transaction.

Exampls incorporating the `arules` package can be found in the Cortana Intelligence Gallery:
- [Frequently Bought Together Product Suggestions via Association Rule Mining](http://gallery.cortanaintelligence.com/Experiment/Frequently-Bought-Together-Product-Suggestions-via-Association-Rule-Mining-1): simulated sales transaction data are mined for association rules, which in turn are used to generate product suggestions for incomplete shopping carts
- [Hai Ning's Association Rules](https://gallery.cortanaintelligence.com/CustomModule/Association-Rules-2): a code-free example introducing the Discover Association Rules custom AML module
- [Martin Machac's Frequently bought together - market basket analyses using ARULES](https://gallery.cortanaintelligence.com/Experiment/Frequently-bought-together-market-basket-analyses-using-ARULES-1): includes a custom R script to load and run the `arules` package from a Script Bundle

## Best Practices for Model Training and Evaluation

### Evaluation Set Creation

To accurately assess a model's likely performance after deployment, the sales data used for evaluation (the "test set") should be as independent as possible from the sales data used for training (the "training set"). If the collaborative filtering data type is used, we must be careful to ensure that data are partitioned at the transaction level: all observations for the same transaction should be included in the same set. (This distinction is automatically held for association rules, in which each transaction is represented by a single row containing a list of products.)

A popular partitioning method is to split the transactions chronologically, mimicking the scenario in which a model was trained with all available data up to a certain date, then tested on data collected after that date. Under this scheme, the model cannot learn and apply trends that appeared only after the training date: its performance during testing will therefore better reflect the performance on brand-new data after deployment. By contrast, if transaction partitioning were random, the model's evaluation performance would be artificially inflated since it could learn time-dependent trends in the test set -- an advantage that will not persist once the model is deployed in the real world.

### Hyperparameter Fitting

It is common practice to reserve a fraction of the training set (either statically by creating a "validation set," or dynamically using cross-validation) to examine the change in the performance of trained models as hyperparameter values are varied. This change in performance may be balanced against the disadvantages of long runtime to select the optimal hyperparameter values.

**Collaborative filtering** algorithms typically take several hyperparameters, including:
- the number of latent dimensions (a larger number may give more accurate results but take longer to train)
- the number of training rounds (a larger number may give more accurate results but take longer to train)
- the number of batches over which to perform training (influences training time)

The common hyperparameters used in **association rules** algorithms are:
- minimum permissible support (low-support rules have little evidence and are more likely to be spurious)
- minimum permissible confidence (low-confidence rules are more often broken, so represent weaker trends)
- maximum item-set size (considering rules involving 3+ items will increase runtime)

### Evaluation

**Collaborative filtering**

One method for evaluating a collaborative filtering (or hybrid recommender) model is to predict ratings for transaction-product pairs using the trained model, and compare these predictions to the actual ratings using metrics like Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and the Pearson correlation coefficient. For Matchbox Recommenders implemented in Azure Machine Learning Studio, this form of evaluation can be performed using the Evaluate Recommender module.

**Association Rules**

Association rule mining typically yields a collection of rules, each of which can be evaluated separately on the test set. For each rule, the support and confidence on the withheld data set is determined. Large deviations of these metrics from the values calculated on the training set may indicate that a change of hyperparameters is appropriate.

## Operationalization and Deployment

### Creating Predictive Web Services from Trained Models

Once optimal hyperparameters have been identified and the model's performance has been evaluated, a final model is fitted using all available data. A web service must be constructed around this model to accept as input a description of an incomplete shopping cart, and provide "frequently-bought-together" suggestions as output. The web service must gracefully handle cases when the shopping cart is empty, new products are mentioned, recommended products are no longer in the catalog, and/or no recommendation can be made. Web services can be deployed with a single click from Azure Machine Learning Studio, or developed in other environments such as Azure App Services. A few model-specific considerations for web service development are included below.

**Collaborative Filtering**
 The "item-to-item" recommendation mode of the collaborative filtering algorithm can then be used to provide one or more product recommendations based for each product currently in a shopping cart. Any duplicates or already-selected products can be removed from the list of recommendations before the recommendations are provided as output. Most collaborative filtering algorithms are able to return an arbitrary number of suggestions per product: this parameter can be tuned to decrease the probability that no suggestion can be provided.

**Association Rules**
Once the final ruleset has been identified by performing association rule mining on all available data, a predictive experiment should be created to identify applicable rules for an input shopping cart and provide as output any suggested products that are not already in the cart.

After the web service has been deployed and tested, the online retailer's website can be modified to call the web service and include responses in the web pages it displays. Many retailer websites are created using common programming languages that support calling and parsing responses from web services. Contoso Mart's website was created on [Azure Web Apps](https://azure.microsoft.com/en-us/services/app-service/web/), which supports many programming languages, including Node.js, Python, PHP, .NET, and Java. Contoso Mart adapts the web service's sample code snippet so that their web app will request a product recommendation each time a user loads a webpage, and incorporate that recommendation into the displayed page.

### A/B Testing <a name="ab"></a>

Before providing a new or upgraded feature to all customers, some online retailers prefer to expose the feature to a subset of users for a trial period. This gradual roll-out diminishes any disruptions that may be experienced as the load handling and real-world recommendation efficacy are first tested. This approach is often called *A/B testing*, since the clickthrough rates and revenue can be compared between the two groups of users that see different site versions. Many retailers randomly assign users to these groups, but when this approach is not properly explained to users, they may express concern or dismay that they do not experience the website in the same way. Some online retailers therefore establish opt-in programs where participants are informed that they may experience features before general availability.

### Model Retraining

Retraining the "frequently-bought-together" model on a scheduled basis will ensure that product suggestions are updated to reflect changes in the product catalog. Programmatic model retraining is available for models created in Azure Machine Learning Studio as well as for the Recommendations API. If desired, A/B testing can be used to phase in new models after every deployment.
