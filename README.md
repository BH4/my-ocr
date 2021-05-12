# my-ocr
Optical character recognition for English uppercase letters, lowercase letters, and digits 0 to 9.

Using "NIST Handprinted Forms and Characters Database" 2nd edition.

Current models are able to reach around 90% accuracy on a merged model in which some of the classes are combined into merged classes. The following list of tuples shows which classes are merged. Each of these merged classes includes an uppercase letter and its lowercase equivalent with the only addition being that the class containing the letter O also contains the number zero.

('c', 'C'), ('k', 'K'), ('m', 'M'), ('p', 'P'), ('s', 'S'), ('u', 'U'), ('v', 'V'), ('w', 'W'), ('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('o', 'O', '0')
