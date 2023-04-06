# **ASSOCIATION RULE LEARNING**
Association rule learning or mining is a rule-based machine learning technique used to discover patterns and relationships within data. This technique is also known as market basket analysis and can be used in both online and offline businesses. It is commonly used as the foundation of recommendation systems in e-commerce. It calculates the probability of products being purchased together frequently and provides recommendations based on those probabilities. Many algorithms are used in rule-based learning. Most used algorithms: Apriori Algorithm, Eclat Algorithm and FP Growth Algorithm.

![image](https://user-images.githubusercontent.com/64617036/230338399-136f9844-f063-49a6-a60a-c02d36217146.png)


## **Apriori Algorithm**
It is a method of basket analysis. It is a statistical algorithm used to reveal the associations of products. It is a statistical evaluation method that reveals product associations with three basic metrics used in this algorithm. It calculates possible product pairs based on a support value determined at the beginning of the study and creates a final table by making eliminations according to the support value determined in each iteration. For example, if the support value of a product is less than 20%, it ignores it while examining the product pairs, for example, if it is less than 20% probability.

Apriori is given by R. Agrawal and R. Srikant in 1994 for frequent item set mining and association rule learning. https://en.wikipedia.org/wiki/Association_rule_learning

Although the Apriori algorithm is the most preferred algorithm for association rules, it scans the database at every stage while calculating support values for object clusters, as in these two algorithms. This leads to an increase in both the processing time and the transaction size. https://www.datasciencearth.com/birliktelik-kurallari-algoritmalari/


Formulae for support, confidence and lift for the association rule X ⟹ Y.
<img width="705" alt="Ekran Resmi 2023-04-06 01 22 55" src="https://user-images.githubusercontent.com/64617036/230338543-85a40c11-4022-43b6-aee1-1537ff1d3963.png">

https://www.researchgate.net/figure/Formulae-for-support-confidence-and-lift-for-the-association-rule-X-Y_fig1_337999958


### **Business Problem**
Below are the cart information of 3 different users. Recommend the most suitable product using association rule for these cart information. Product recommendations can be one or more. Derive decision rules based on 2010-2011 German customers.

* Product id in User 1's cart: 21987
* Product id in User 2's cart: 23235
* Product id in User 3's cart: 22747

### **About Dataset**
The dataset named Online Retail II includes online sales transactions of a UK-based retail company between 01/12/2009 and 09/12/2011. The company's product catalog includes gift items, and it is known that most of its customers are wholesalers.

* InvoiceNo: Invoice Number (If this code starts with C, it indicates that the transaction has been canceled)
* StockCode: Product code (Unique for each product)
* Description: Product name
* Quantity: Number of products sold in the invoices
* InvoiceDate: Invoice date
* UnitPrice: Invoice price (in pounds)
* CustomerID: Unique customer number
* Country: Country name
