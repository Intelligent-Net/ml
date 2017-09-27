This is a Java solution to this contest :

    https://www.kaggle.com/dalpozz/creditcardfraud

Make sure you download creditcard.csv from the site above

run as :

    java uk.co.inet.card.CreditCard creditcard.csv

or 

    java uk.co.inet.card.CreditCardAnomaly creditcard.csv

The first version is a self contained linear regression sparse solution. The 
anomaly version finds outliers. Both versions use correlation to find the
features that correlate well with frauds.

There are many possible extensions to these algorithms.

chris@intelligent-net.co.uk
