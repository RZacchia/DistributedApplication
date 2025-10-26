# Advanced Distributed Systems - Exercise 01 - Robert Zacchia

## Application BookRent Overview

I choose to build a book rental portal for this course. A quick summary of the different tasks of the application:
- editing and browsing the cataloge of available books
- renting and return books, and keeping count
- add books that are not available to a watchlist
- different user roles (Anonymous, Customer, Employee, Administrator)
- Authentification and Autorisation

To fullfill the aformentioned tasks I decided to make the application out of 5 services in total:
 - Gateway
 - Cataloge
 - User
 - Identity
 - Renting