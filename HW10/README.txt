## Structure
HW10/
├── data/
│   └── videoplayback.mp4
│
├── results/
│   ├── kcf/
│   ├── csrt/
│   └── comparison_plot.png
│
├── tracker_compare.py
├── README.md
└── requirements.txt


	Comparison
- Do you see any differences?

Yes, clear differences are observed.

KCF tracker
Very stable bounding box
Minimal changes across frames
Less sensitive to object movement
CSRT tracker
Adaptive bounding box
Changes size and position
Follows object more precisely

- Does one tracker perform better than the other?

Both trackers successfully tracked the object in all frames:
KCF: 15/15
CSRT: 15/15

However:
KCF
Faster
More stable
Less accurate
CSRT
More accurate
Better handles scale changes
Slightly less stable

	Visualization

Final Conclusion
CSRT performs better in terms of accuracy and adaptability
KCF performs better in terms of speed and stability
CSRT is preferable for dynamic scenarios, while KCF is suitable for simpler tracking tasks.
