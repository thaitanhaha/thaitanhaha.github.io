---
title: 'Reinforcement learning 1 - Q-Learning'
date: 2025-06-30
permalink: /posts/reinforcement-learning-1/
excerpt: 'Q-Learning là một thuật toán học tăng cường nền tảng, giúp các tác nhân học được hành động tốt nhất cần thực hiện trong một môi trường nhất định bằng cách tối đa hóa phần thưởng theo thời gian.'
tags:
  - reinforcement learning
---

Học tăng cường là một lĩnh vực con trong học máy, trong đó một *tác nhân* học cách tối ưu hóa *hành động* của mình trong một *môi trường* để đạt được mục tiêu lâu dài. 

Khác với các phương pháp học có giám sát và không giám sát, học tăng cường **không có** tập dữ liệu đã được gán nhãn.

- Tác nhân (Agent): Là người đưa ra quyết định. 

- Môi trường (Environment): Môi trường có thể thay đổi theo các hành động của tác nhân.

- Hành động (Action): Những hành động mà tác nhân có thể thực hiện để tương tác với môi trường. 

- Trạng thái (State): Đại diện cho tình trạng hiện tại của môi trường tại một thời điểm cụ thể, bao gồm tất cả thông tin mà tác nhân cần để quyết định hành động tiếp theo.

- Phần thưởng (Reward): Mỗi hành động của tác nhân trong môi trường sẽ nhận một phần thưởng hoặc hình phạt. 

- Chính sách (Policy): Chiến lược cho phép tác nhân chọn hành động ở mỗi trạng thái.

> **Ví dụ:** Trong trò chơi CartPoleBalance, *tác nhân* là người điều khiển chiếc xe; *môi trường* là chiếc xe và cột; các *hành động* là di chuyển xe sang trái hoặc di chuyển xe sang phải; *trạng thái* bao gồm vị trí của xe, vận tốc của xe, góc của cột, vận tốc góc của cột; *phần thưởng* có thể được định nghĩa là +1 mỗi lần tác nhân giữ cột đứng trong một bước và +0 nếu cột ngã quá 15 độ.

Mục tiêu của học tăng cường là giúp tác nhân học được một chính sách tối ưu, tức là một chiến lược cho phép tác nhân chọn hành động tốt nhất ở mỗi trạng thái để tối đa hóa tổng phần thưởng tích lũy theo thời gian. Tức là

1. Tác nhân bắt đầu ở một trạng thái cụ thể.
2. Dựa trên trạng thái hiện tại, tác nhân chọn một hành động để thực hiện.
   - Khám phá (exploration): Tác nhân thử những hành động *mới*.
   - Khai thác (exploitation): Tác nhân tận dụng những hành động *đã biết* để đạt được phần thưởng cao nhất.
3. Môi trường thay đổi trạng thái theo hành động của tác nhân và trả về phần thưởng.
4. Tác nhân nhận phần thưởng và cập nhật chiến lược hành động của mình.

## Q - learning

Một vấn đề quan trọng trong học tăng cường là **cập nhật chiến lược như thế nào**. Q-learning là một cách để giải quyết câu hỏi này. 

- Q-value (Q-value table): Giá trị dự đoán của tác nhân về phần thưởng mà nó sẽ nhận được khi thực hiện một hành động trong một trạng thái cụ thể. Mỗi **cặp trạng thái và hành động** có **một giá trị Q** được lưu trong bảng.

<p align="center">
  <img src="https://www.researchgate.net/profile/Manel-Abdellatif/publication/361359070/figure/fig1/AS:1167899863584769@1655460434256/Cart-Pole-balancing-problem.png" alt="CartPoleBalance">
</p>


Với bài toán CartPoleBalance trên, Q-table được viết như thế nào? Giả sử ta chia mỗi thành phần của trạng thái thành 6 khoảng, tổng số trạng thái rời rạc là 6 (vị trí xe) × 6 (vận tốc xe) × 6 (góc của cột) × 6 (vận tốc góc của cột) = 1296 trạng thái.

| Trạng thái (vị trí xe, vận tốc xe, góc của cột, vận tốc góc của cột)   | Hành động 0 (di chuyển trái) | Hành động 1 (di chuyển phải) |
| ------------ | ------------ | ------------ |
| (0, 0, 0, 0) | 0.01         | 0.04         |
| (0, 0, 0, 1) | 0.03         | 0.07         |
| (..., ..., ..., ...) | ...         | ...         |
| (5, 5, 5, 5) | 0.02         | 0.05         |

Nghĩa là, ví dụ khi gặp trạng thái (0,0,0,1), hành động được chọn là di chuyển sang phải bởi vì nó cho phần thưởng trung bình cao nhất. 

Vậy cập nhật như thế nào để có các số như 0.01, 0.04,... trong bảng? Các giá trị ban đầu của Q-table thường được gán là 0 hoặc một giá trị ngẫu nhiên. Sau đó, công thức cập nhật Q-value là:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right)
$$

Trong đó:

* \\(Q(s, a)\\): Giá trị Q hiện tại của cặp trạng thái \\(s\\) và hành động \\(a\\).
* \\(\alpha\\): Tốc độ học (learning rate), giúp điều chỉnh mức độ cập nhật Q-value.
* \\(r\\): Phần thưởng nhận được sau khi thực hiện hành động \\(a\\) tại trạng thái \\(s\\).
* \\(\gamma\\): Hệ số chiết khấu (discount factor), dùng để điều chỉnh sự quan trọng của các phần thưởng trong tương lai.
* \\(\max_{a'} Q(s', a')\\): Giá trị Q tối đa của các hành động có thể thực hiện tại trạng thái tiếp theo \\(s'\\).

Trong học tăng cường, có hai khái niệm là khám phá (exploration) và khai thác (exploitation), là cách mà tác nhân chọn một hành động để thực hiện. Để chọn hành động, thuật toán sử dụng chính sách epsilon-greedy. Nghĩa là, với Q-learning, với xác suất \\(1−\epsilon\\), tác nhân chọn hành động có giá trị Q cao nhất (khai thác); với xác suất \\(\epsilon\\), tác nhân chọn hành động ngẫu nhiên (khám phá) để khám phá môi trường.

**Tóm lại**, để bắt đầu học tăng cường, khởi đầu từ một trạng thái \\(\rightarrow\\) ta mô phỏng môi trường và hành động của tác nhân \\(\rightarrow\\) cập nhật bảng Q \\(\rightarrow\\) tiếp tục mô phỏng và cập nhật cho đến khi kết thúc trò chơi, hoặc đến số bước nhất định. 

## Code
Code để dễ hình dung nhé!

<details><summary markdown="span">Môi trường, Tác nhân</summary>

```python
class CartPoleEnv:
    def __init__(self):
        # Trạng thái: [x, x_velocity, theta, theta_velocity]
        # Hành động: 0 di chuyển sang trái, 1 di chuyển sang phải
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.g = 9.8
        self.m = 0.1
        self.M = 1.0
        self.L = 0.5
        self.dt = 0.02
        
    def step(self, action):
        # Hàm dùng để mô phỏng
        x, x_velocity, theta, theta_velocity = self.state
        
        force = 20.0 if action == 1 else -20.0
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        total_mass = self.M + self.m
        pole_mass_length = self.m * self.L
        
        temp = (force + pole_mass_length * theta_velocity ** 2 * sin_theta) / total_mass
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (self.L * (4 / 3 - self.m * cos_theta ** 2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        x += x_velocity * self.dt
        x_velocity += x_acc * self.dt
        theta += theta_velocity * self.dt
        theta_velocity += theta_acc * self.dt
        
        lose = x < -2.4 or x > 2.4 or theta < -np.pi / 15 or theta > np.pi / 15
        reward = 1.0 if not lose else 0.0
        
        self.state = np.array([x, x_velocity, theta, theta_velocity])
        
        return self.state, reward, lose

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state
        
class QLearningAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(state_space + (action_space,))
        self.state_bins = [
            np.linspace(-2.4, 2.4, 10), # vị trí của xe
            np.linspace(-3.0, 3.0, 10), # vận tốc của xe
            np.linspace(-np.pi / 15, np.pi / 15, 10), # góc của cột
            np.linspace(-3.0, 3.0, 10), # vận tốc góc của cột
        ]

    def discretize_state(self, state):
        # Chia trạng thái (hiện là giá trị liên tục) thành các khoảng rời rạc
        discretized = []
        for i in range(len(state)):
            discretized.append(np.digitize(state[i], self.state_bins[i]) - 1)
        
        return tuple(discretized)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon: # Khám phá
            return np.random.choice(self.action_space)
        else: # Khai thác
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        # Cập nhật theo công thức
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state + (best_next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_error
```

</details>

<details><summary markdown="span">Huấn luyện</summary>

```python
def train(agent, env, episodes, max_timesteps=200):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = agent.discretize_state(state)
        total_reward = 0
        
        for t in range(max_timesteps):
            action = agent.choose_action(state)
            next_state, reward, lose = env.step(action)
            next_state = agent.discretize_state(next_state)
            
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if lose: break
        
        rewards.append(total_reward)
        if episode % 500 == 0 or episode == episodes-1:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return rewards
```

</details>

<details><summary markdown="span">Kiểm thử</summary>

```python
def evaluate(agent, env, episodes, is_render):
    for episode in range(episodes):
        state = env.reset()
        state = agent.discretize_state(state)
        lose = False
        total_reward = 0
        
        while not lose:
            # Với kiểm thử, chỉ dùng Khai thác chứ không dùng Khám phá
            action = np.argmax(agent.q_table[state])
            state, reward, lose = env.step(action)
            state = agent.discretize_state(state)
            total_reward += reward
            
            if is_render: render(env)
        
        print(f"Episode {episode}, Total Reward: {total_reward}")
```

</details>

## Tại sao?

Ta đã biết công thức cập nhật của Q: lấy giá trị Q cũ, cộng thêm một phần nhỏ của chênh lệch giữa phần thưởng tương lai dự kiến  và giá trị Q cũ. Nhưng tại sao người ta lại dùng công thức này?

### Quy trình ra quyết định Markov (Markov Decision Process - MDP)

Trong trí tuệ nhân tạo, người ta dùng thuật ngữ `Quy trình ra quyết định Markov (Markov Decision Process - MDP)` để mô hình hóa các tình huống mà trong đó các quyết định được đưa ra liên tiếp và kết quả của các hành động là không chắc chắn. 

> Tính chất Markov là tính chất mà tức là tương lai chỉ phụ thuộc vào hiện tại, và không phụ thuộc vào quá khứ.

$$
MDP = (S, A, P, R, \gamma)
$$

* \\(S\\): Tập trạng thái.
* \\(A\\): Tập hành động.
* \\(P(s' \mid s, a)\\): Xác suất chuyển từ trạng thái \\(s\\) sang trạng thái \\(s'\\) khi thực hiện hành động \\(a\\).
* \\(R(s, a)\\): Phần thưởng nhận được khi thực hiện hành động \\(a\\) tại trạng thái \\(s\\). 
* \\(\gamma\\): Hệ số chiết khấu.

Để giải được MDP, người ta thường dùng `Quy hoạch động (Dynamic Programming - DP)` dựa trên `Phương trình Bellman`.

Trong Q-learning, ta không cần phải biết mô hình môi trường (bao gồm các xác suất chuyển trạng thái \\(P(s' \mid s, a)\\)) khi cập nhật Q-values. Thay vào đó, Q-learning chỉ cần quan sát trạng thái, hành động, phần thưởng và trạng thái tiếp theo trong quá trình tương tác với môi trường là được. Do đó, ta có thể "tạm quên đi" \\(P(s' \mid s, a)\\).

### Phương trình Bellman cho Giá trị Trạng thái (Value Function)

Phương trình Bellman cho **giá trị trạng thái** \\(V(s)\\) được viết như sau:

$$ V(s) = \max_a \left(R(s,a) + \gamma V(s') \right) $$

### Phương trình Bellman cho Giá trị Hành động (Q-value Function)

Trong phương trình Bellman cho giá trị trạng thái, chúng ta quan tâm đến tất cả các trạng thái và tất cả các hành động khả thi. Vậy khi bỏ hàm \\(\max\\), chúng ta sẽ được công thức như là giá trị của một trạng thái được tạo ra cho chỉ một hành động khả thi.

Dựa trên ý tưởng đó, phương trình Bellman áp dụng cho **giá trị hành động** được viết như sau:

$$ Q(s, a) = R(s, a) + \gamma V(s') $$

Để tạo nên sự đồng nhất, ta viết lại \\(V(s')\\) bằng \\(\max_{a'} Q(s', a')\\) vì chúng ta coi giá trị của một trạng thái được tính bằng giá trị lớn nhất có thể của hành động \\(Q(s, a)\\). 

$$ Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a') $$

### Chênh lệch thời gian (Temporal Difference - TD)

Môi trường không phải bất biến, vậy phải làm như nào để nắm bắt được sự thay đổi của môi trường? Rất đơn giản, lấy giá trị Q mới trừ giá trị Q cũ!

$$ \alpha TD(s,a) = Q(s, a)_{new} - Q(s,a)_{old} $$

Vậy

$$ Q(s, a)_{new} \leftarrow Q(s, a)_{old} + \alpha TD(s,a) = Q(s, a)_{old} + \alpha \left( r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)_{old} \right) $$

## Sự hội tụ

Một điều đặc biệt, Q-learning là thuật toán đầu tiên được chứng minh hội tụ đến chính sách tối ưu (optimal policy) dưới một số điều kiện nhất định. Nó được đề xuất ban đầu bởi Watkins 1989 và được chứng minh hội tụ bởi Watkins & Dayan 1992. 

Nói chung, phải đáp ứng hai điều kiện để đảm bảo sự hội tụ (chính sách sẽ trở nên gần với chính sách tối ưu một cách tùy ý sau một khoảng thời gian dài tùy ý):

- Tốc độ học phải tiến tới 0, nhưng không quá nhanh. 

    Về mặt hình thức, điều này đòi hỏi tổng tỷ lệ học phải phân kỳ, nhưng tổng bình phương của chúng phải hội tụ. 

    $$\sum_{t}^{\infty} \alpha_t = \infty \quad \quad\sum_{t}^{\infty} \alpha^2_t < \infty$$

    > Ví dụ: `1/1, 1/2, 1/3, 1/4, ...`

- Mỗi cặp trạng thái - hành động phải được truy cập vô hạn lần. 

    Nghĩa là, về mặt toán học, mỗi hành động trong mọi trạng thái phải có xác suất được chính sách chọn > 0. Trong thực tế, sử dụng chính sách epsilon-greedy (\\(\epsilon > 0\\)) đảm bảo rằng điều kiện này được đáp ứng.

Dưới các điều kiện này, Q-learning sẽ hội tụ tới giá trị tối ưu trong thời gian vô hạn, nhưng trong thực tế, thường có một số yếu tố như độ chính xác của mô phỏng, việc dừng quá sớm, hoặc tốc độ học không phù hợp có thể ảnh hưởng đến quá trình hội tụ.

## Nhận xét

- Về mặt lý thuyết, Q-Learning có thể hội tụ về giải pháp tối ưu nhưng thường không rõ ràng về cách điều chỉnh các siêu tham số như \\(\epsilon\\) và \\(\alpha\\).

- Khi số lượng trạng thái và hành động tăng lên, số lượng giá trị Q cần lưu trữ và cập nhật cũng tăng theo cấp số nhân, làm cho Q-learning gặp phải vấn đề về hiệu suất và bộ nhớ. 

    > Ví dụ trong bài toán CartPoleBalance, thật ra các yếu tố về góc hay vị trí là liên tục, nhưng để giảm bộ nhớ, ta đã chia giá trị liên tục thành rời rạc.

Hãy nhớ rằng Q-learning là một thuật toán cũ và hơi lỗi thời, đây là một cách cơ bản và sơ khai để tìm hiểu về học tăng cường, nhưng có những cách tốt hơn để giải quyết một vấn đề thực tế!

