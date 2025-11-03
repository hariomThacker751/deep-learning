<h1>Transformer Model (PyTorch)</h1>

<p>
This repository contains a single <b>Jupyter Notebook</b> that implements a <b>Transformer model from scratch</b> using <b>PyTorch</b>.  
It is a simple and educational implementation of the ideas from  
<a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> (Vaswani et al., 2017).
</p>

<hr>

<h2>🚀 Overview</h2>
<ul>
  <li>Implementation of core Transformer components:
    <ul>
      <li>Multi-Head Self-Attention</li>
      <li>Positional Encoding</li>
      <li>Encoder and Decoder blocks</li>
      <li>Feed Forward Networks</li>
    </ul>
  </li>
  <li>Step-by-step code explanation inside the notebook</li>
  <li>Training on a toy dataset (sequence-to-sequence example)</li>
</ul>

<hr>

<h2>🧠 Architecture</h2>

<pre>
Input → Embedding + Positional Encoding → Encoder → Decoder → Output
</pre>

<p>Each block includes Self-Attention, Add & Norm, Feed Forward, and Residual Connections.</p>

<hr>

<h2>📦 Requirements</h2>

<p>Make sure you have Python 3.8+ and install the following packages:</p>

<pre>
pip install torch numpy matplotlib
</pre>

