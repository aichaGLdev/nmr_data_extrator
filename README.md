# nmr_data_extractor

**nmr_data_extractor** is a project focused on the automatic extraction of spectroscopic data from Nuclear Magnetic Resonance (NMR) spectra images. By combining advanced **computer vision techniques** with **supervised learning algorithms** such as **KNN** and **SVM**, the system enables the detection, recognition, and interpretation of key information from both 1D and 2D spectra. The processing pipeline includes **image preprocessing**, **optical character recognition (OCR)**, and **object detection**, facilitating the structural elucidation of small organic molecules.

## Features

- Automatic extraction of data from 1D and 2D NMR spectra images.
- Integration of computer vision and supervised learning techniques (KNN, SVM).
- Image preprocessing, OCR, and object detection for accurate data interpretation.
- Structural elucidation of small organic molecules.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository/nmr_data_extractor.git
    ```
2. Navigate to the project folder:
    ```bash
    cd nmr_data_extractor
    ```
3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
6. Run the Flask application:
    ```bash
    flask run
    ```

## Usage

- Once the Flask app is running, you can interact with the project via the API or UI.
- The backend processes NMR spectra images uploaded by the user and returns the extracted data for structure elucidation.

## Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.
