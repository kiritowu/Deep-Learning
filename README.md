<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Deep Learning</h3>

  <p align="center">
    What we want is a machine that can learn from experience. <cite>— Alan Turing</cite>
    <br />
    <br />
    <br />
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Repository

This repository serves as an archive of  my learning journey and projects that I have embarked in the realm of Deep Learning and its applications.

Feel free to contact me if you have any queries or spot any mistakes in my implementation.

### Table of Contents

This section list out the projects in this repository.
<table>
    <tr>
        <th>Subfields</th>
        <th>Project Title</th>
        <th>Descriptions</th>
        <th>Keywords</th>
    </tr>
    <tr>
        <td rowspan=2>CV</td>
        <td><a href="https://github.com/kiritowu/Deep-Learning/tree/main/CV/EfficientNetV2_with_Image_Classification_Benchmarks">EfficientNetV2 with Image Classification Benchmarks</a></td>
        <td>Utilizing SOTA model and training procedure on classic Image classification dataset like Fashion-MNIST and CIFAR10 using Timm running on Pytorch.</td>
        <td>
            <em><b>Image Classification</b></em>,<br>
            <em><b>EfficientNetV2</b></em>,<br>
            <em><b>RandAugment</b></em>,<br>
            <em><b>Progressive Learning</b></em>,<br>
            <em><b>Timm</b></em>,<br>
            <em><b>Pytorch</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td><a href="https://github.com/kiritowu/Deep-Learning/tree/main/CV/FasterRCNN_VehiclesDetection">Faster-RCNN Vehicles Detection</a></td>
        <td>Detecting cars using Faster-RCNN with MobilenetV3 as backbone.</td>
        <td>
            <em><b>Object Detection</b></em>,<br>
            <em><b>Faster-RCNN</b></em>,<br>
            <em><b>MobileNetV3</b></em>,<br>
            <em><b>FPN</b></em>,<br>
            <em><b>Timm</b></em>,<br>
            <em><b>Pytorch</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td rowspan=3>GAN</td>
        <td><a href="https://github.com/kiritowu/Deep-Learning/tree/main/GAN/AC-BIGGAN-with-CIFAR10">AC-BIGGAN with CIFAR10</a></td>
        <td>Generating small coloured images with AC-BIGGAN.</td>
        <td>
            <em><b>ACGAN</b></em>,<br>
            <em><b>BIGGAN</b></em>,<br>
            <em><b>Conditional Batch Normalization</b></em>,<br>
            <em><b>Hinge Loss</b></em>,<br>
            <em><b>Label Smoothing</b></em>,<br>
            <em><b>IS</b></em>,<br>
            <em><b>FID</b></em>,<br>
            <em><b>Pytorch</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td><a href="https://github.com/kiritowu/GDL_code">GDL_code</a></td>
        <td>Forked repository while reading O'Reilly's Generative Deep Learning book.</td>
        <td>
            <em><b>VAE</b></em>,<br>
            <em><b>WGAN</b></em>,<br>
            <em><b>WGANGP</b></em>,<br>
            <em><b>CycleGAN</b></em>,<br>
            <em><b>Keras</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td><a href="https://github.com/kiritowu/Generative-Adversarial-Networks-Projects">Generative-Adversarial-Networks-Projects</a></td>
        <td>Forked repository while reading PacktPublishing's Generative Adversarial Networks Projects book.</td>
        <td>
            <em><b>3DGAN</b></em>,<br>
            <em><b>cGAN</b></em>,<br>
            <em><b>DCGAN</b></em>,<br>
            <em><b>SRGAN</b></em>,<br>
            <em><b>StackGAN</b></em>,<br>
            <em><b>CycleGAN</b></em>,<br>
            <em><b>Keras</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td rowspan=3>RL</td>
        <td><a href="https://github.com/kiritowu/Deep-Learning/tree/main/RL/Great-Lunar-Lander-with-DQN">Great Lunar Lander with DQN</a></td>
        <td>Solving Lunar-Landerv2 with DQNs approach.</td>
        <td>
            <em><b>DQN</b></em>,<br>
            <em><b>DDQN</b></em>,<br>
            <em><b>SARSA</b></em>,<br>
            <em><b>OpenAI Gym</b></em>,<br>
            <em><b>Keras</b></em><br>
        </td>
    </tr>
    <tr>
        <td><a href="https://github.com/kiritowu/Hands-On-Reinforcement-Learning-with-Python">Hands-On Reinforcement Learning with Python</a></td>
        <td>Coding exercise and self-made notes from the Hands-On Reinforcement Learning with Python book published by Packt.</td>
        <td>
            <em><b>Reinforcement Learning</b></em>,<br>
            <em><b>Markov Decision Process</b></em>,<br>
            <em><b>Monte Carlo Control</b></em>,<br>
            <em><b>Otw</b></em>,<br>
        </td>
    </tr>
    <tr>
        <td><a href="https://github.com/kiritowu/Deep-Learning/tree/main/RL/Snake-DQN">Snake-DQN</a></td>
        <td>Solving Snake with DQN approach.</td>
        <td>
            <em><b>DQN</b></em>,<br>
            <em><b>Keras</b></em><br>
        </td>
    </tr>
</table>


## Prerequisites

The list of standard Python3 packages that I have used for my Machine Learning projects is shown in `requirements.txt`.
To install all of the packages specific to each subfields, simply call the following command:
* pip
  ```sh
  cd CV
  pip install -r requirements.txt
  ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Zhao Wu Wong, Bryan - [@LinkedIn](https://www.linkedin.com/in/zhao-wu-wong-27b434201/) - zhaowu.wong@gmail.com

Kaggle Profile: [https://www.kaggle.com/kiritowu](https://www.kaggle.com/kiritowu)

<!-- Credits -->
## Credits

- Cars Object Detection Dataset: [https://www.kaggle.com/sshikamaru/car-object-detection](https://www.kaggle.com/sshikamaru/car-object-detection)
- README Template : [https://github.com/othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)
