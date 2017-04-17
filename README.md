# Personalized Offers Data Science Playbook

Mary Wahl, Ph.D.<br/>
*Data Scientist at Microsoft*

## Executive Summary


Offer personalization is the selection of displayed advertisements based on customers' personal characteristics and actions. Personalized offers may be used to inform customers of relevant products in large online marketplaces, entice visitors to continue browsing, and remind users of their intended purchases. This playbook, intended for data scientists and other industry professionals, covers best practices for the complete data science process in three common use cases. Each use case follows fictitious online retailer Contoso Mart from data acquisition to implementation and refinement of a solution.

## Use Cases

### Personalized Offers using Classifier Models

Displayed web pages can be customized to include an advertisement deemed most likely to be clicked by a given user. High-quality predictions may leverage records of user behaviors like browsing history and purchases, as well as semi-static attributes like demographics and interests. The example highlighted in this use case is based on the [Cortana Intelligence Solution How-to Guide](https://github.com/Azure/Cortana-Intelligence-Suite-Industry-Solutions) for [Personalized Offers](https://github.com/Azure/cortana-intelligence-personalized-offers), a deployment-ready solution demonstrating the integration of machine learning with other Azure services.

Click [here](https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/Personalized_Offers_from_Classifiers_Use_Case.md) to learn more about this use case.

### Product Suggestions using Hybrid Recommender Models

Many online retailers offer a rotating selection of thousands of products. Hybrid recommender models, which combine collaborative filtering and content-based filtering, are well-suited for serving product selections based on a customer's interests and past behavior under these conditions. This use case includes a [Cortana Intelligence Gallery](https://gallery.cortanaintelligence.com/) example illustrating the use of a Matchbox Recommender model.

Click [here](https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/Product_Suggestions_Using_Hybrid_Recommenders.md) to learn more about this use case.

### "Frequently Bought Together" Product Suggestions

In addition to product recommendations based on longterm user attributes like demographics and interests, many retailers provide suggestions based on users' current shopping cart contents. These suggestions highlight items that are usually purchased at the same time as the already-selected product, e.g. a box of nails to complement a new hammer. In this use case, we describe how association rules and item-to-item collaborative filtering models can be used to serve "frequently bought together" product suggestions.

Click [here](https://github.com/Azure/cortana-intellligence-personalization-data-science-playbook/blob/master/Frequently_Bought_Together_Product_Suggestions_Use_Case.md) to learn more about this use case.

## Further Reading

- [Personalized Offers Solution How-to Guide](https://github.com/Azure/cortana-intelligence-personalized-offers) 

    This Cortana Intelligence Solution use case is highlighted in the "Personalized Offers using Classifier Models" use case.
    
    
- [Cortana Industry Solutions Repository](https://github.com/Azure/Cortana-Intelligence-Suite-Industry-Solutions)

    This GitHub repository lists deployment-ready Cortana Industry Solutions for other industry verticals.
