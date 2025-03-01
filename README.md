# XTTS-Runner
A lightweight, easy-to-use library for XTTS-v2 inference

# Setup

* Download the model (git lfs needs to be installed): `git clone https://huggingface.co/coqui/XTTS-v2`
* Install the requirements: `pip install -r requirements.txt`
* Run inference: `python src/synthesize.py ./XTTS-v2/ --text "What does the fox say?" --lang en --speaker "Claribel Dervla" --output_file fox.wav`

### For Streaming

* Install PyAudio: https://pypi.org/project/PyAudio/
* Run inference: `python src/synthesize_streaming.py ./XTTS-v2/ --text "What does the fox say? Ding Ding Ding Ding." --lang en --speaker "Claribel Dervla" --speed 0.6 --chunk_size 20`

### Available Speakers

'Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski'

### Known Issues

- Because of a [PyTorch issue](https://github.com/pytorch/pytorch/issues/134416) with MPS you need a very recent version of torch (e.g. `pip install --pre torch==2.7.0.dev20250226 torchaudio==2.6.0.dev20250226 --index-url https://download.pytorch.org/whl/nightly/cpu`) and macOS 15.1 or later to run generation on longer texts (very short texts still work).
