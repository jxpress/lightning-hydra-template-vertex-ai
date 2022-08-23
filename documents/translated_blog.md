
# Problems and solutions when using hyperparameter tuning in Vertex AI for code written in Hydra
This document is modified and translated into English from [this blog](https://tech.jxpress.net/entry/2022/05/13/113011).
![theme_image](/documents/images/theme.jpeg)

---
Hello, my name is Tanaka and I am an intern on the ML team at JX PRESS Corporation. I have worked on object detection, image matching, text generation, and MLOps. In this blog, I would like to share the difficulties I had trying to run a code written in Hydra with my mentor, Yongtae, a senior ML engineer, on Vertex AI's hyperparameter tuning job, and the solution we found!

## 📎 Introduction

Based on the philosophy of "Focus our power where it should be used" the ML team at JX PRESS Corporation has created template codes for machine learning, including PyTorch Lightning and Hydra.

Please read the explanatory article written by [Yongtae](https://github.com/Yongtae723) about the philosophy and template codes.
[How we at JX PRESS Corporation devise for team development of R&D that tends to become a genus](https://tech.jxpress.net/entry/2021/10/27/160154) and [PyTorch Lightning explained by a heavy user](https://techjxpress.net/entry/2021/11/17/112214).

---
Training AI takes a lot of time, sometimes several hours per training session.

If the training of hyperparameter tuning is run in series on a single machine as in the example above in Figure 1, it will take more than one training time * number of training attempts, which can take several days or more.

![fig_1](/documents/images/fig_1.png)
<p align = "center">
Fig.1 - Comparison of the time required to complete hyperparameter tuning when training on a single machine (top example) multiple machines in parallel (bottom example). The Vertex AI hyperparameter tuning can  parallelize training, but and perform Bayesian optimization by connecting parallel training in series, which enables cost-effective and highly accurate search for the optimal parameters. 
</p>

<details>
<summary><b>More info about optimization methods</b></summary>
Hyperparameter optimize can be divided into the following two main methods

(1) A method in which hyperparameters are determined in advance and experiments are conducted, and the parameter with the best performance is adopted (grid search is a well-known example).

(2) The method of searching for the optimal hyperparameters by conducting an experiment with a certain hyperparameter and, based on the results, conducting another experiment with a certain hyperparameter and repeating the process (Bayesian estimation is a well-known example).

In the case of (1), the hyperparameters to be searched for are determined before the experiment, so all the experiments can be conducted in parallel at once, and can be completed with only the first experiment in the example below in Figure 1. However, since the hyperparameters are chosen on an ad hoc basis, there are many unnecessary calculations, and the cost of obtaining the optimal hyperparameters may be enormous.

On the other hand, method (2) strategically selects hyperparameters, so it can search for optimal parameters cost-effectively and with high precision. However, since the hyperparameters to be explored in the next experiment depend on the previous learning results, it is not possible to run all learning in parallel at once.

If parallel learning could be connected in series, we could benefit from both parallel learning and Bayesian optimization, but this is generally very difficult to implement. However, with Vertex AI, parallel learning can be serialized with very little effort (Figure 1, bottom).
</details>

<br>

One of the values we hold dear at JX PRESS Corporation is Speed, and in order to support this philosophy from the ML side as well, we want to finish training faster and speed up the development for our business.

Therefore, we would like to run many experiments in parallel when optimizing hyperparameters. Such parallel training can be easily achieved using Vertex AI, a fully managed ML service (see example below in Fig.1).

On the other hand, the hyperparameter tuning function of Vertex AI Training is incompatible with Hydra, which is used in the template code of JX PRESS Corporation, and we could not use it as it is.

## 😖 Problem.
In hyperparameter tuning, the Vertex AI passes hyperparameters to the train containers using command line arguments, and after learning is complete, optimization of the hyperparameters is performed by sending the metric values to the Vertex AI from trian containers.

However, as shown below, the command line argument format passed by Vertex AI and the corresponding command line argument format by Hydra are different, so I could not get it to work.

<br>


### The format described in the official Vertex AI documentation
Vertex AI recommended the format of the `argparse` like below
```bash
python3 -m my_trainer --learning_rate learning-rate-in-this-trial
```

<aside>⚠️ Attention.

As far as I do, the actual format being passed is not the one above, but rather

--learning_rate=learning-rate-in-this-trial

The recommended argparse format is also used.

The recommended argparse will also accept this format.
</aside>

<br>


### The format of Hydra
```bash
python3 my_trainer.py learning_rate=learning-rate-in-this-trial
```

## 🧪 Possible solutions
There are several possible solutions.

1. Wait for Vertex AI to support the Hydra format

    → Best, but not sure if it will be supported.

2. Use Optuna, an open source framework for automatic hyperparameter optimization, instead of using Vertex AI's hyperparameter tuning function.

    → Using Hydra+Optuna in series, not parallel, and complete in one instance is convenient. However, when running it by multiple machines, it is difficult because it is necessary to set up a SQL server. ([About Optuna parallelization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html))

3. Rewrite command line argument analysis library from Hydra to argparse.

    → It is hard, and it is difficult to define a parser, and config files cannot be used.

4. Insert a process of converting the format of arguments between Vertex AI and training container.

    → This is aggressive, but the lowest cost to implement.

In this repository, we adopted the least expensive option 4.

## 🐉 How to convert command line arguments into format of Hydra
The implemented code is [vertex_ai/scripts/hparams_tuning/convert_command.sh](/vertex_ai/scripts/hparams_tuning/convert_command.sh).


Here, we use shell script to convert command line arguments.In shell script, the first argument is taken as `$1`, and the n-th argument is taken as `$n`. In this case, the number of arguments is variable, so `$@` is used to receive the entire argument as a string.

Next, the received string is converted to Hydra format by regular expression substitution.

Since this is a shell script, the `sed` command is used to perform the regular expression substitution.

`--key=value` and `--key value` can be both accepted by setting following command.

```bash
sed -r 's/--([^= ]*)[= ]([^ ]*)/\1=\2/g'
```

The sed command is similar to Vim's replace command, with `/` delimited by `s/regex/substitute/flag`, where *(expr)* corresponds to `\1, \2` and the last g is a flag for multiple substitutions. The `g` at the end is the flag for multiple substitutions. Also, the sed command does not support shortest match, and it handles it by repeating all characters except the one after the one you want to match. (`[^ ]*` places, for example).

- example
```bash
❯ echo "--lr=0.001 --batch-size 64" | sed -r 's/--([^= ]*)[= ]([^ ]*)/\1=\2/g'

lr=0.001 batch-size=64
```

The original training code can be executed using the converted arguments.

Since *$(expr)* accepts the result of executing *expr* as a string, you can use it to add the converted arguments after the original learning code execution command by typing the following.
```bash
python train.py $(echo $@ | sed -r 's/--([^= ]*)[= ]([^ ]*)/\1=\2/g') # train.pyはHydraを用いたコード
```

Finally, create a script file with this as train.sh, etc., and include it in the Vertex AI [config file](/vertex_ai/configs/hparams_tuning/default.yaml) execution command as `./convert_command.sh`

## 💡 Result
As a result of these efforts, Vertex AI was able to compute hyperparameter tuning in parallel (Figs.2~4). In Figs.3 and 4, there are multiple colored lines at the same time, which means that multiple instances are being trained in parallel.

**This allowed me to finish in a few hours what used to take me several days to learn!**

![fig_2](/documents/images/fig_2.png)
<p align = "center">
Fig.2 - Each trial of hyperparameter tuning job and the hyperparameter values used in that trial. Specific variable names are hidden.
</p>

![fig_3](/documents/images/fig_3.png)
<p align = "center">
Fig.3 - CPU utilization transitions during hyperparameter tuning for Vertex AI. The color of each line corresponds to each trial.
</p>

![fig_4](/documents/images/fig_4.png)
<p align = "center">
Fig.4 - GPU utilization transitions during hyperparameter tuning for Vertex AI. The color of each line corresponds to each trial.
</p>


## 📝 
Conclusion

During the Vertex AI hyperparameter tuning job, we were able to convert the hyperparameters passed from Vertex AI into a format that Hydra could accept by interrupting the script and using regular expressions to convert the parameter format.

This allowed us to adjust the hyperparameters in parallel while using the existing code with Hydra, thus reducing the learning time.

There may be a better way other than this method. I am still in the process of learning, so any advice on how to improve would be greatly appreciated.

※This blog was written by Tanaka and edited and translated by his mentor, Yongtae, a senior ML engineer.