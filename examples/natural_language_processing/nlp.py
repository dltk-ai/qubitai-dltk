import dltk_ai

client = dltk_ai.DltkAiClient('YOUR_APIKEY')

# sentiment analysis
sentiment_text ="Thank you so much for referring your friend Jack to us. I've enjoyed getting to know them and do business with them. I'm so happy that you've stuck around with us for this long and brought your friend to share the experience with you. We're lucky to have you. Thanks again for being such a fantastic customer!"
response = client.sentiment_analysis(sentiment_text)
print(response)

# sentiment analysis using azure and ibm_watson
response = client.sentiment_analysis(sentiment_text, sources = ['azure','ibm_watson'])
print(response)

# parts-of-speech tagger
pos_text = 'And now for something completely different'
response = client.pos_tagger(pos_text)
print(response)
# parts-of-speech tagger - ibm_watson & spacy
response = client.pos_tagger(pos_text, sources = ['ibm_watson','spacy'])
print(response)

# named entity recognition
ner_text = 'He moved to California to work for Google.'
response = client.ner_tagger(ner_text)
print(response)

# named entity recognition - using azure & ibm_watson
response = client.ner_tagger(ner_text, sources = ['azure','ibm_watson'])
print(response)

# dependency parser
dependency_parser_text = "Artificial intelligence, is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality. The distinction between the former and the latter categories is often revealed by the acronym chosen."
response = client.dependency_parser(dependency_parser_text)
print(response)

# tags
tags_text = "TweetDeck is a social media dashboard application for management of Twitter accounts. Originally an independent app, TweetDeck was subsequently acquired by Twitter Inc. and integrated into Twitter's interface."
response = client.tags(tags_text)
print(response)




