# 🩺 Dr. Hubert's Medication, Reaction & Severity Analyzer

An AI-powered adverse drug reaction (ADR) detection system that combines GPT-4o, FDA data, and specialized machine learning models to analyze patient descriptions and identify potential medication-related adverse events.

## What It Does

Dr. Hubert's ADR Analyzer processes natural language patient descriptions to extract medications and adverse reactions, then cross-references this information with FDA databases to identify potentially novel or severe adverse drug reactions. The system provides healthcare professionals with automated ADR detection and severity scoring to support clinical decision-making.

### Key Features

- **Natural Language Processing**: Extract medications and reactions from free-text patient descriptions
- **GPT-4o Integration**: Advanced AI extraction with handling for typos, casing issues, and medical terminology variations
- **FDA Data Integration**: Real-time comparison with openFDA adverse event database
- **Severity Scoring**: ML-based ADR severity prediction using specialized models
- **Novel Reaction Detection**: Identifies reactions not previously associated with specific medications
- **Clinical Decision Support**: Provides structured analysis for healthcare professionals

## Try It Live

**[Launch the App on Hugging Face Spaces](https://huggingface.co/spaces/UVA-MSBA/M4_Team4_ADR)**

## How It Works

1. **Text Extraction**: GPT-4o analyzes patient descriptions to extract medication names and adverse reactions
2. **FDA Database Query**: Retrieves known adverse reactions for each medication from openFDA API
3. **Fuzzy Matching**: Compares extracted reactions with known reactions using similarity algorithms
4. **Novel Reaction Detection**: Identifies reactions not previously documented for specific medications
5. **Severity Assessment**: Uses paragon-analytics/ADRv1 model to calculate ADR severity scores
6. **Clinical Summary**: Generates structured analysis report for healthcare review

## Technical Stack

- **Python**: Core programming language
- **GPT-4o**: Natural language extraction and medical entity recognition
- **OpenFDA API**: FDA adverse event database integration
- **Transformers**: Hugging Face model integration (paragon-analytics/ADRv1)
- **PyTorch**: Deep learning framework for severity prediction
- **Gradio**: Web interface for clinical users

## Analysis Components

### Severity Scoring
- Uses specialized ADR classification model
- Provides probability scores for severe adverse reactions
- Trained on clinical adverse event data

### Medication Extraction
- Handles various medication name formats
- Accounts for brand names, generic names, and common misspellings
- Filters out non-medical references

### Reaction Analysis
- Compares patient-reported reactions with FDA database
- Identifies potentially novel adverse events
- Uses fuzzy matching for terminology variations

## Clinical Use Cases

- **Pharmacovigilance**: Automated ADR detection and reporting
- **Clinical Decision Support**: Real-time medication safety assessment
- **Research**: Identification of novel adverse drug reactions
- **Patient Safety**: Early warning system for severe reactions
- **Drug Development**: Post-market surveillance support

## Usage Examples

### Sample Patient Description
```
"Patient has been taking Lipitor 20mg daily for 6 months. 
Recently developed severe muscle pain and weakness in both legs. 
Also experiencing occasional nausea and fatigue. 
Blood work shows elevated liver enzymes."
```

### Expected Output
- **Extracted Medications**: Lipitor
- **Extracted Reactions**: muscle pain, weakness, nausea, fatigue, elevated liver enzymes
- **Severity Score**: 0.75 (high severity)
- **Known vs Novel Reactions**: Comparison with FDA database

## Privacy & Security

- **No Data Storage**: Patient descriptions are processed in real-time without persistent storage
- **API Security**: Secure communication with OpenAI and FDA APIs
- **HIPAA Considerations**: Designed for clinical environments with appropriate safeguards

## Target Users

- **Healthcare Professionals**: Physicians, pharmacists, nurses
- **Clinical Researchers**: ADR surveillance teams
- **Regulatory Affairs**: Pharmaceutical safety officers
- **Healthcare IT**: Clinical decision support system developers

## Future Enhancements

- **Multi-language Support**: Processing patient descriptions in multiple languages
- **Integration APIs**: Direct EHR system integration
- **Advanced Analytics**: Temporal pattern analysis and trend detection
- **Mobile Interface**: Tablet-optimized interface for clinical use
- **Batch Processing**: Bulk analysis capabilities for research

## Contributing

This project was developed as part of the UVA MSBA program. Contributions from healthcare professionals and data scientists are welcome.

## License

This project is intended for educational and research purposes. Please consult with legal and compliance teams before clinical deployment.

## Acknowledgments

- **OpenAI**: GPT-4o language model capabilities
- **FDA**: openFDA API for adverse event data
- **Paragon Analytics**: ADRv1 severity classification model
- **UVA MSBA Program**: Academic support and guidance
- **Healthcare Partners**: Clinical validation and feedback

## Important Disclaimer

This tool is designed to assist healthcare professionals and should not replace clinical judgment. All results should be reviewed by qualified medical personnel before making clinical decisions.
