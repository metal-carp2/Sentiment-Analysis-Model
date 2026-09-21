
import dependencies.audiodata as aud
import matplotlib.pyplot as plt
import numpy as np

def show_live_debug_display(dataset: aud.AudioDataSet):
    print()
    for i in dataset.ranges:
        if len(i) > 0:
            print(i.dialogue + "  " + f"[{i[0].timestamp} - {i[len(i)-1].timestamp}]")

    timestamps = [round(item.timestamp, 2) for item in dataset]
    n = max(1, len(timestamps) // 20)  # Adjust n dynamically based on the number of points
    pitches = [item.pitch for item in dataset]
    volumes = [item.volume for item in dataset]
    timestamps_labels = [str(ts) for ts in timestamps]

    pitchSampleRate = 0.4
    count = 1
    temp = []
    newPitches = []
    for i in range(len(pitches)):
        temp.append(pitches[i])
        if timestamps[i] > pitchSampleRate*count or i>=len(pitches)-1:
            count+=1
            for x in temp:
                newPitches.append(np.mean(temp))
            temp = []
    pitches = newPitches

    fig, ax1 = plt.subplots()
    bar_width = 0.4
    x = np.arange(len(timestamps))  # X positions

    pitch_bars = ax1.plot(x - bar_width / 2, pitches, color="blue", label="Pitch")
    ax1.set_ylabel("Pitch", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(x[::n])  # Show x-axis ticks only for every nth point
    ax1.set_xticklabels([timestamps_labels[i] for i in range(0, len(timestamps_labels), n)], rotation=45, ha="right")
    ax2 = ax1.twinx()
    volume_bars = ax2.plot(x + bar_width / 2, volumes, color="orange", label="Volume")
    ax2.set_ylabel("Volume", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    plt.title("Pitch and Volume at Timestamps with Dual Y-Axes")
    fig.tight_layout()

    plt.show()