# MoviePlotGPT

This project is inspired by [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on building a GPT model from scratch. The goal is to scale the initial idea given by the tutorial and achieve better text generations with different datasets.

This model does not have a special tokenizer. We just encode every character to an integer. So it's a really simple implementation but even scaling this can lead to really impressing results.

## Dataset
At first it was trained on [The Office Lines Dataset](https://docs.google.com/spreadsheets/d/18wS5AAwOh8QO95RwHLS95POmSNKA2jjzdt0phrxeAE0/edit?gid=747974534#gid=747974534). It contains all every line from every episode and it's a really clean dataset. You can access it by clicking the link or from my repo. I just converted the csv file to a txt with `file_formatter.py`.

Although it was just close 4 million characters, it did produce some incredible results. You can check them out from the `outputs/office` folder.

I decided to scale the data and used the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). It has plot summaries of 34886 movies around the world. It was way bigger than the office dataset. I converted this one to a txt too. But this time every movie got added to the txt with this format:

`The plot summary of the movie named '<MovieTitle>' is:<MoviePlot>END_OF_PLOT`

I thought this would help the model a lot to understand the patterns, so it can generate movies from their titles somehow.



## Project Details
- **Model Architecture**: I experimented with various model sizes, including different numbers of layers, heads, and embedding dimensions (`n_embd`).
- **Datasets**:
  - The **Office Lines Dataset** can be found under `outputs/office`.
  - The **Movie Plots Dataset** consists of 34k movies and was used for more extensive training.

## Weights and Model Sizes
I experimented with different model sizes and configurations to understand how they affect the quality of generated plots. The final models were trained using various hyperparameters, and the weights will be shared for both datasets:

- **Office Lines Model Weights**: (Link to be added)
- **Movie Plots Model Weights**: (Link to be added)

You can access and use these weights to generate your own plot summaries.

## How to Use This Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MoviePlotGPT.git
   cd MoviePlotGPT
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pretrained Weights**:
   - Use the provided download script to fetch the pretrained weights. Detailed instructions and download links will be provided in the `weights/README.md` file.

4. **Run the Model**:
   - You can generate plot summaries using the provided `generate.py` script.
   - Example:
     ```bash
     python generate.py --input "The plot summary of the movie named 'Inception' is:"
     ```

## Future Improvements
- Further fine-tuning with larger datasets.
- Experimenting with different training strategies, tokenizers (e.g., Byte-Pair Encoding), and subword tokenization techniques to improve coherence.
