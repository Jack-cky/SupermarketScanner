<!-- INTRODUCTION -->
<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/banner.png"></a>
</div>

<br>In early April, two doctors were found guilty of stealing food valued at over HKD 1,600 from the Purple Label supermarket in Hong Kong ([The Standard](https://www.thestandard.com.hk/breaking-news/section/4/202373/Two-top-doctors-each-fined-HK$5,000-for-stealing-food-from-AEON-in-Whampoa), 2023). Although they claimed to have forgotten to scan the items at the self-checkout due to distractions, their selective scanning behaviour suggested otherwise, leading the magistrate to dismiss their defence. Unfortunately, retail theft involving self-checkouts has become a pervasive issue for retailers, resulting in significant financial losses.

To address this problem, we propose an innovative solution called SupermarketScanner. This system scans all items placed on the self-checkout counter, automatically recognising the sale price and the total number of units in the basket. This not only streamlines the checkout process for customers but also helps retailers prevent future theft incidents by reinforcing their self-checkout systems and ensuring that all products are scanned before customers leave the store.

**First Published:** 17 April 2023  
**Last Updated:** 25 June 2025


<!-- ROADMAP -->
## Table of Contents
- [1 - Objective](#1)
- [2 - Transfer Learning with Online Images](#2)
- [3 - Evaluating Model Effectiveness](#3)
- [4 - Practical Implementation](#4)
    - [4.1 - Single Item per Transaction](#4.1)
    - [4.2 - Multiple Items per Transaction](#4.2)
- [5 - Tackling Model Limitations](#5)
- [6 - Fine-Tuning with Offline Images](#6)
- [7 - Reflections on Feasibility](#7)


<!-- SECTION 1 -->
<a name="1"></a>

## Objective: Simplify Checkout and Prevent Shoplifting
SupermarketScanner is an AI-driven system integrated into the self-checkout counter, designed to recognise products placed on the counter. Inspired by [BakeryScanner](https://www.a-1bakery.com.hk/en/news/detail.html?CMS_FRONT_INFO_ID=344), it is engineered to be fast, accurate, and efficient, utilising a You Only Look Once (YOLO) model.

With SupermarketScanner, you simply place your items on the self-checkout counter, and the system automatically detects the products, displaying their prices and a transaction summary. This eliminates the need for manual barcode scanning, streamlining the checkout process and saving valuable time for customers. For retailers, it helps prevent instances where customers might "forget" to scan items and later "claim distractions."

Bid farewell to long queues and embrace hassle-free checkout with SupermarketScanner—your ultimate shopping companion.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/demo.gif" width="70%"></a>
</div>


<!-- SECTION 2 -->
<a name="2"></a>

## Transfer Learning with Online Images
This project employs a pre-trained YOLO v8 model that has been trained on the COCO dataset. However, this model is not directly applicable to our specific domain. To address this limitation, we need to gather our own images by searching online and apply transfer learning to enhance our supermarket product recognition system. As a proof of concept (PoC) project, we selected eight common items: blueberries, bread, chicken, eggs, juice, melon, sushi, and watermelon, collecting 55 images for each item.

To label these images, we utilised [Roboflow](https://universe.roboflow.com/jack-chan-edpdi/supermarketscanner), which provides a user-friendly interface. Due to the limited number of images, it is challenging to train a decent model that can accurately detect items. Therefore, we applied augmentation to the training data using various image transformations, effectively tripling the number of training examples.

<div align="center">
  <a href="https://universe.roboflow.com/jack-chan-edpdi/supermarketscanner"><img src="./imgs/image_annotation.gif" width="70%"></a>
</div>


<!-- SECTION 3 -->
<a name="3"></a>

## Evaluating Model Effectiveness
Behind the scenes, we experimented with multiple methods to train the model. We discovered that models trained with augmented images generally outperformed those without image augmentation. Our final model achieved impressive results, with 87% mAP50 (mean average precision at IoU 0.5) on the development (dev) set and 93% on the test set. The evaluation graphs below show promising signs of the potential for implementing the model in real-world applications. We cannot wait to see how it performs at the self-checkout counter.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/model_performance.png" width="70%"></a>
</div>


<!-- SECTION 4 -->
<a name="4"></a>

## Practical Implementation
It is time to put SupermarketScanner into practical application. We selected a range of products for testing: blueberries, bread, eggs, juice, and sushi. These items were arranged on the desk to simulate the checkout process.

<a name="4.1"></a>

### Single Item per Transaction
We were immediately impressed by SupermarketScanner's swift recognition of each item, accurately displaying its price and transaction summary without delay. However, our confidence in the YOLO model was somewhat shaken when it struggled to recognise the blueberries during checkout.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/application_single.gif" width="70%"></a>
</div>

<a name="4.2"></a>

### Multiple Items per Transaction
The ability to recognise and process multiple items simultaneously is crucial for our application. SupermarketScanner effectively demonstrated its capacity to detect a variety of products, even those with similar shapes or packaging. It quickly and seamlessly processed multiple items, providing evidence of its potential to reduce wait times and enhance customer satisfaction during checkout.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/application_multiple_success.gif" width="70%"></a>
</div>

However, things are not as perfect as they seem. Another set of transactions revealed areas for improvement. For instance, the system struggled to identify products positioned at different angles and those obscured by other items. This is not ideal for retailers in a production environment, highlighting the need to enhance the model before introducing the tool to businesses.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/application_multiple_failure.gif" width="70%"></a>
</div>


<!-- SECTION 5 -->
<a name="5"></a>

## Tackling Model Limitations
As part of the iterative process in developing SupermarketScanner, several problems were identified during this trial. Below, we highlight key issues and propose solutions to address them.

1. **Misclassification of Background Noise**: At times, the system misclassifies background noise as an object.
    - Solution: Remove obstructions from the background or select a table with a single-colour background to minimise distractions.

2. **Failure Related to Rotated Items**: the model's performance suffers when it encounters rotated items.
    - Solution: Implement a higher degree of rotation in image augmentation to ensure that the model is better equipped to handle various orientations.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/environment_setting.png" width="70%"></a>
</div>

3. **Class Imbalance**: A class imbalance leads to the model's inability to accurately identify certain classes.
    - Solution: Collect additional images from underrepresented classes to achieve a more balanced data distribution.

4. **Insufficient Training Images**: There are not enough images available to train a more robust model.
    - Solution: Source more images through web scraping and by capturing real transaction images at the self-checkout counter. It is vital to carefully select these images to ensure they are relevant and representative of the target population while avoiding any biases in the model.

5. **Non-Identical Data Distribution**: The data distribution is not uniform across the train-test and deployment sets, resulting in discrepancies.
    - Solution: Acquire images from the self-checkout counter and organise them into a dev-test set. Verify that this dev-test set accurately reflects the real test environment (the self-checkout counter) and evaluate the model on multiple test sets to ensure its generalisability.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/imbalance_distribution.png" width="70%"></a>
</div>

By addressing these issues, we can enhance the effectiveness of SupermarketScanner and ensure its successful implementation in a retail environment.


<!-- SECTION 6 -->
<a name="6"></a>

## Fine-Tuning with Offline Images
SupermarketScanner was initially trained using online images, which differ significantly from the environment in our specific use case. To enhance the system's ability to detect items on the self-checkout counter, we needed to incorporate images captured directly from the counter. We sacrificed transactions involving single products, as demonstrated in the previous section, to fine-tune the YOLO model. Meanwhile, we manually synthesised additional images with various rotations from those videos. The remaining six transaction samples were then used to evaluate the overall model performance.

We are excited about the results, which show that the fine-tuned model significantly outperforms the initial version, demonstrating a higher capability to detect items on the table. As shown below, it can now correctly identify blueberries and eggs, which the initial model struggled to detect. However, the system still fails to identify overlapping items on the desk. This limitation arises because we did not expose the model to such scenarios during training and fine-tuning, making it understandable that the system struggles in these cases. This highlights an area for further improvement in the future.

<div align="center">
  <a href="https://supermarketscanner.streamlit.app/"><img src="./imgs/fine_tune_improvement.gif" width="70%"></a>
</div>


<!-- SECTION 7 -->
<a name="7"></a>

## Reflections on Feasibility
SupermarketScanner is a project that leverages technology to address the real-life problem of retail theft at self-checkout counters. While this project serves as a PoC, it is important to note that the YOLO model we trained for this purpose is not yet ready for deployment in a production environment.

Our aim is to demonstrate a potential solution to this issue, but can SupermarketScanner truly address the problem? As a former employee at the Red Label supermarket in Hong Kong, I believe it may not be entirely feasible. Unlike BakeryScanner, which has only around 200 SKUs ([The Standard](https://www.thestandard.com.hk/breaking-news/section/4/195048/A-1-Bakery-announces-technological-breakthroughs-for-the-benefit-of-customers), 2022), most supermarkets stock thousands of items. The time and cost required to train the model and implement an AI camera system could be significant. Moreover, our experiences with BakeryScanner in their stores reveal similar issues, such as the misclassification of background noise and types of bread. However, cashiers can immediately correct these errors. In contrast, SupermarketScanner operates as a self-service machine, meaning that if the system misrecognises an item, it could lead to false alarms and upset customers, resulting in an unpleasant checkout experience.

So, is SupermarketScanner entirely useless? Not quite. It has the potential to be used as a back-end system to double-check whether customers have scanned all their items before completing their transactions, thereby helping supermarkets prevent self-checkout theft. For example, if a pair of customers forget to scan items valued at over HKD 1,600, SupermarketScanner can alert security or staff to investigate further. Although SupermarketScanner may not streamline the checkout process, it can still play a valuable role in mitigating incidents of shoplifting.

<div align="center">
  <a href="https://www.thestandard.com.hk/breaking-news/section/4/202373/Two-top-doctors-each-fined-HK$5,000-for-stealing-food-from-AEON-in-Whampoa">
    <img src="./imgs/reenactment.png" width="70%">
  </a>
</div>


<!-- MISCELLANEOUS -->
<a name="8"></a>

## Product Backlog
This project is managed using a product backlog. You can review the [backlog](https://docs.google.com/spreadsheets/d/1hZBngU6REh5M9iyUclPlf8IyO3Iz3ZVW1exo_-vM1ks/pubhtml?gid=103533477&single=true) to understand the prioritised list of features, changes, enhancements, and bug fixes planned for future development.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details. Feel free to fork and collaborate on the project!
