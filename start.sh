# start.sh
#!/bin/bash

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Run the Flask app
python app.py
