# FIT5230 Project: Certifying LLM Safety against Adversarial Prompting 🛡️ defending LLM Safety\!

Welcome to our repository for FIT5230\! 🎉🎉🎉 We're tackling the exciting challenge of making Large Language Models (LLMs) safer against clever attacks.

## 🚀 Introduction

Modern LLMs are fine-tuned to be helpful and harmless. If you ask them to do something bad, they should politely decline.

Here's an example of a well-behaved, aligned LLM saying "no" to a harmful prompt:

<p align="center">
  <img src="figures/harmful_prompt.png" width="500"/>
</p>


But what if someone tries to trick the LLM? 😈 These "adversarial attacks" add a special, sneaky sequence of words to a harmful prompt, tricking the LLM into generating unsafe content. It's like a secret password to bypass the safety rules\!

Check out how a simple attack can break the safety alignment:

<p align="center">
  <img src="figures/adversarial_attack.png" width="500"/>
</p>


These attacks can even be automated using algorithms like GCG, creating an endless supply of jailbreaks. 😱

To fight this, we present **Erase-and-Check** a *certified* defense that provides a verifiable safety guarantee against these attacks. Our method ensures that if our safety filter is good at spotting clean harmful prompts, it will be just as good at spotting them even when they're under attack\! 💪

### How does it work? 💡

Our procedure is simple but effective:

1.  It takes the input prompt and starts **erasing** tokens one by one.
2.  It then **checks** each of these shorter subsequences with a safety filter.
3.  If *any* of the subsequences (or the original prompt) are flagged as harmful, the entire prompt is labeled as harmful. ✅

We use two types of safety filters:

1.  🤖 A general-purpose LLM like **Llama 2**.
2.  🧠 A fine-tuned **DistilBERT classifier** trained on examples of safe and harmful prompts.

### Attack Modes We Studied ⚔️

We tested our defense against three types of attacks:

1.  **Adversarial Suffix:** The sneaky sequence is added at the end of the prompt. `[Harmful Prompt] + [Attack Sequence]`
2.  **Adversarial Insertion:** The attack sequence is inserted somewhere in the middle. `[Part 1] + [Attack Sequence] + [Part 2]`
3.  **Adversarial Infusion:** Attack tokens are sprinkled anywhere in the prompt. `[H..a..r..m..f..u..l]`

Here’s a cool visual of Erase-and-Check in action\!

\<p align="center"\>
\<img src="figures/erase-and-check.png" width="700"/\>
\</p\>

## 📂 What's In This Repository?

Here's a map of our project files:

  * `defenses.py` 🛡️: Implements our core Erase-and-Check logic and the safety filters.
  * `main.py` 🚀: The main script to run all our experiments.
  * `data/` 📝: Contains the safe and harmful prompts for training and testing.
  * `safety_classifier.py` 🏋️: The script to train our DistilBERT safety classifier.
  * `models/` 🧠: Where we store our trained classifier models.
  * `results/` 📊: All the juicy results from our experiments are saved here as JSON files and plots.
  * `greedy_ec.py` & `grad_ec.py` ⚡: Implementations of our faster empirical defenses.
  * `gcg.py` 👾: Our implementation of the GCG attack to test our defenses.
  * `bash scripts/` 📜: Super handy scripts to reproduce our main results quickly\!

**A quick note:** To run our code, you'll need access to a GPU. We used a powerful NVIDIA A100 for all our experiments\! 💻

## ✅ Our Certified Accuracy Guarantee

Here's the best part: we can mathematically prove that the accuracy of Erase-and-Check on *attacked* harmful prompts is at least as high as the safety filter's accuracy on *clean* harmful prompts. This means we don't even need to run attacks to know our minimum accuracy\!

Here are the certified accuracy scores:

  * **Llama 2-based filter**: **\~92%** 🏆
  * **DistilBERT-based filter**: **99%** 🥇

Want to verify this yourself?

```bash
# For the Llama 2 filter
python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt

# For the DistilBERT filter
python main.py --num_prompts 120 --eval_type harmful --use_classifier --model_wt_path models/[model-weights-file].pt --harmful_prompts data/harmful_prompts_test.txt
```

## ⚔️ Attack Mode 1: Adversarial Suffix

Here's how our two defenses perform on **safe prompts** under a suffix attack. We want the accuracy to be high (not misclassifying safe as harmful) and the run time to be low\!

\<p align="center"\>
\<img src="results/comparison\_safe\_suffix\_acc.png" width="300" style="margin-right: 20px;"/\>
\<img src="results/comparison\_safe\_suffix\_time.png" width="300"/\>
\</p\>

👉 Use `bash scripts/jobs_suffix.sh` to get these results\!

## ⚔️ Attack Mode 2: Adversarial Insertion

Now for the insertion attack\! Here's the performance comparison on safe prompts:

\<p align="center"\>
\<img src="results/comparison\_safe\_insertion\_acc.png" width="300" style="margin-right: 20px;"/\>
\<img src="results/comparison\_safe\_insertion\_time.png" width="300"/\>
\</p\>

👉 Use `bash scripts/jobs_insertion.sh` to get these results\!

## ⚔️ Attack Mode 3: Adversarial Infusion

And finally, the tricky infusion attack\! Performance on safe prompts:

\<p align="center"\>
\<img src="results/comparison\_safe\_infusion\_acc.png" width="300" style="margin-right: 20px;"/\>
\<img src="results/comparison\_safe\_infusion\_time.png" width="300"/\>
\</p\>

👉 Use `bash scripts/jobs_infusion.sh` to get these results\!

## 🏋️‍♂️ Training Our Safety Classifier

We trained our DistilBERT classifier on the safe and harmful prompts in the `data/` directory. To make it robust, we also included the erased subsequences of safe prompts in the training data, teaching the model that these fragments are also safe.

To train your own classifier, use this command:

```bash
python safety_classifier.py --safe_train data/safe_prompts_train_[mode]_erased.txt --safe_test data/safe_prompts_test_[mode]_erased.txt --save_path models/distilbert_[mode].pt
```

Or just run our handy script: `bash scripts/train_safety_clf.sh`\!

Don't want to wait? You can download our pre-trained models from this [Dropbox link](https://www.dropbox.com/scl/fi/ux4ew8y88uslu5064r2xh/models.zip?rlkey=4bo1njpnj4nc801tw1pkby52o&dl=0). 🎁

## ⚡️ Efficient Empirical Defenses

Our certified method checks *everything*, which can be a bit slow. So, we also developed three faster "empirical" defenses. They don't come with a mathematical guarantee, but they are very effective in practice\!

1.  **RandEC** 🎲: Randomly samples and checks a subset of the erased subsequences.
2.  **GreedyEC** 🤔: Greedily erases tokens that are most likely to be part of a harmful phrase.
3.  **GradEC** 📈: Uses gradients to smartly decide which tokens to erase.

We tested them against the GCG attack. Below are the results\!

### RandEC Performance

\<p align="center"\>
\<img src="results/empirical\_suffix\_120\_clf\_rand.png" width="400"/\>
\</p\>
👉 Run `bash scripts/jobs_rand_ec.sh` to reproduce.

### GreedyEC Performance

\<p align="center"\>
\<img src="results/greedy\_ec\_120\_clf.png" width="400"/\>
\</p\>
👉 Run `bash scripts/jobs_greedy_ec.sh` to reproduce.

### GradEC Performance

\<p align="center"\>
\<img src="results/grad\_ec\_120\_clf.png" width="400"/\>
\</p\>
👉 Run `bash scripts/jobs_grad_ec.sh` to reproduce.

## 🛠️ Get Set Up\! Installation Guide

Ready to run the code? Follow these steps to set up your environment.

1.  **Install Anaconda** 🐍

      * Download the installer from [anaconda.com](https://www.anaconda.com/products/distribution).
      * Run the installer script in your terminal.

2.  **Create a Conda Environment** 🌱

    ```bash
    conda create -n fit5230_env python=3.10
    ```

3.  **Activate the Environment** ✅

    ```bash
    conda activate fit5230_env
    ```

4.  **Install PyTorch with CUDA** 🔥

      * Head over to the [PyTorch website](https://pytorch.org/) for the command specific to your system. It should look something like this:

    <!-- end list -->

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

5.  **Install Huggingface Transformers** 🤗

    ```bash
    conda install -c huggingface transformers
    ```

6.  **Install Accelerate** 🚀

    ```bash
    conda install -c conda-forge accelerate
    ```

7.  **Install Scikit-learn** 🧠

    ```bash
    conda install -c anaconda scikit-learn
    ```

8.  **Install Seaborn** 📊

    ```bash
    conda install anaconda::seaborn
    ```

You should be all set\! Happy coding and thanks for checking out our project\! 😊
