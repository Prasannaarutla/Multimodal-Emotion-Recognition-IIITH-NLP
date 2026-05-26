import matplotlib.pyplot as plt


speech_accuracy = [

    18.08,
    39.20,
    60.45,
    73.75,
    88.44,
    91.79,
    93.48,
    98.44,
    93.08,
    98.88,
    99.29,
    98.08,
    98.08,
    99.06,
    98.79,
    99.20,
    98.79,
    99.38,
    99.20,
    99.78
]


speech_loss = [

    70.0448,
    48.5463,
    32.8571,
    21.8916,
    13.3150,
    7.9344,
    6.3571,
    3.6714,
    8.2328,
    2.7027,
    1.6664,
    2.5371,
    2.7403,
    1.7765,
    1.4953,
    1.0735,
    1.8888,
    1.1962,
    1.3064,
    0.6682
]


text_accuracy = [

    12.95,
    13.44,
    13.75,
    13.35,
    14.11,
    13.44,
    14.29,
    14.15,
    14.51,
    14.20
]


text_loss = [

    1102.2539,
    1099.3227,
    1096.3597,
    1097.5182,
    1095.1613,
    1095.3853,
    1094.9029,
    1094.0754,
    1094.9411,
    1093.6294
]


epochs_speech = range(
    1,
    21
)

epochs_text = range(
    1,
    11
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_speech,
    speech_accuracy,
    marker='o'
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("Speech Pipeline Accuracy")

plt.savefig(
    "Results/plots/speech_accuracy.png"
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_speech,
    speech_loss,
    marker='o'
)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Speech Pipeline Loss")

plt.savefig(
    "Results/plots/speech_loss.png"
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_text,
    text_accuracy,
    marker='o'
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("Text Pipeline Accuracy")

plt.savefig(
    "Results/plots/text_accuracy.png"
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_text,
    text_loss,
    marker='o'
)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Text Pipeline Loss")

plt.savefig(
    "Results/plots/text_loss.png"
)

plt.close()


print("\nPlots Generated Successfully!")

models = [

    "Speech",

    "Text"
]


accuracies = [

    99.78,

    14.20
]


plt.figure(figsize=(7, 5))

plt.bar(
    models,
    accuracies
)

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.title("Model Performance Comparison")

plt.savefig(
    "Results/plots/model_comparison.png"
)

plt.close()