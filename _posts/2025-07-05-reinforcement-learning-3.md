---
title: 'Reinforcement learning 3 - Policy Gradient'
date: 2025-07-05
permalink: /posts/reinforcement-learning-3/
excerpt: 'Policy Gradient cho phép mô hình học trực tiếp chính sách hành động thông qua tối ưu hóa gradient. Kỹ thuật này đặc biệt hiệu quả trong các bài toán có không gian hành động liên tục hoặc phức tạp.'
tags:
  - reinforcement learning
---

Nhắc lại về Q-learning và Deep Q-learning, chúng vẫn tuân theo chính sách nhất định

  $$
  \pi(s) = 
  \begin{cases} 
  \text{argmax}_a Q(s, a) & \text{với xác suất } 1 - \epsilon \\
  \text{random action} & \text{với xác suất } \epsilon 
  \end{cases}
  $$

Có thể thấy, việc tối ưu hóa hàm giá trị Q không trực tiếp tối ưu chính sách hành động (gọi là **off-policy**). Hơn nữa, đôi khi phần thưởng không xuất hiện liên tục, nên việc điều chỉnh giá trị Q trở nên rất khó khăn.

Và quan trọng nhất, chúng chỉ hoạt động tốt trong không gian hành động rời rạc.

<p align="center">
  <img src="https://pylessons.com/media/Tutorials/Reinforcement-learning-tutorial/Beyond-DQN/PG_vs_DQN.png" alt="Policy Gradient vs Deep Q-learning">
</p>

## Policy Gradient

Đầu tiên, ta tham số hóa một chính sách ngẫu nhiên (stochastic policy). Cũng có nghĩa là, một mạng nơ-ron đưa ra một phân phối xác suất các hành động. Vậy thì mục tiêu là: tìm các tham số \\(\theta\\) để điều chỉnh phân phối các hành động này sao cho hành động đem lại nhiều phần thưởng nhất thì có xác suất chọn cao nhất \\(\rightarrow\\) tối đa hóa phần thưởng mong đợi. 

> Ta tối ưu hóa chính sách trực tiếp nên gọi là **on-policy**.

Ý tưởng là: Trong một tập (episode), nếu thắng thì **mỗi** hành động đã thực hiện là tốt, tăng xác suất xảy ra của chúng; nếu thua thì **mỗi** hành động đã thực hiện là không tốt, giảm xác suất xảy ra của chúng.

Do đó, hàm phần thưởng hay hàm mục tiêu được viết như là: phần thưởng tích lũy kỳ vọng trên một quỹ đạo (trajectory - dãy trạng thái và hành động).

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi} \left[ R(\tau)\right] = \sum_{\tau} \textcolor{blue}{P(\tau;\theta)} \textcolor{purple}{R(\tau)}$$ 

- \\(P(\tau;\theta)\\): Xác suất mỗi quỹ đạo có thể xảy ra (xác suất này phụ thuộc vào \\(\theta\\))
	
    > Có thể hiểu rằng, nếu \\(\theta\\) thay đổi \\(\rightarrow\\) chính sách thay đổi \\(\rightarrow\\) cách tác nhân chọn hành động trong từng bước sẽ khác \\(\rightarrow\\) các trạng thái được ghé thăm sẽ khác \\(\rightarrow\\) các hành động được chọn tiếp theo cũng sẽ khác \\(\rightarrow\\) xác suất của quỹ đạo \\(\tau\\) cũng sẽ thay đổi.

	$$ \textcolor{blue}{P(\tau;\theta)} = \mu(s_0) \prod_{t=0}^{T} P(s_{t+1} \mid s_t, a_t) \pi_\theta (a_t \mid s_t)$$
    
    trong đó \\(\mu(s_0)\\) là phân phối trạng thái ban đầu, \\(\pi_\theta(a_t \mid s_t)\\) là xác suất chính sách chọn hành động \\(a_t\\) từ trạng thái \\(s_t\\).

- \\(R(\tau)\\): Phần thưởng tích lũy từ một quỹ đạo bất kỳ.

	$$ \textcolor{purple}{R(\tau)} = \sum_{t=0}^{T} \gamma^t r_t $$

Để tối đa hóa \\(J(\theta)\\) thì rất tự nhiên, ta dùng **gradient ascent**,

$$ \theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$$

Tuy nhiên, có hai vấn đề khi tính đạo hàm của \\(J(\theta)\\):

1. Để đạo hàm, ta cũng cần biết đạo hàm của hàm phân phối trạng thái \\(P(s_{t+1} \mid s_t, a_t)\\), gọi là MDP dynamics. Điều này gắn liền với môi trường. Ta không lấy đạo hàm được vì ta có thể không biết về nó.

	> Ở Policy Gradient, ta tối ưu hóa chính sách trực tiếp, nên không thể *bỏ qua* hàm phân phối trạng thái như Q-learning được.

    \\(\Rightarrow\\) Người ta chứng minh được rằng đạo hàm của hàm mục tiêu **không liên quan** đến đạo hàm của hàm phân phối trạng thái, gọi là **Policy Gradient Theorem**. 

	$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \left[ \textcolor{darkorange}{\nabla_\theta \log  \pi_\theta (a_t \mid s_t) R(\tau)}\right]\right] $$ 
    
2. Không thể tính toán chính xác đạo hàm của hàm mục tiêu vì nó yêu cầu tính xác suất của mỗi quỹ đạo có thể xảy ra \\(\rightarrow\\) rất tốn kém về mặt tính toán. 
	
    \\(\Rightarrow\\) Ta ước lượng đạo hàm bằng cách sử dụng *một tập* quỹ đạo \\(\mathcal{D} = \\{\tau\\}_{i=1,2,\dots,N}\\), gọi là **phương pháp Monte-Carlo**.
    
    $$ \nabla_\theta J(\theta) \approx \dfrac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \left[ \textcolor{darkorange}{\nabla_\theta \log  \pi_\theta \left(a_t^{(i)} \mid s_t^{(i)}\right) R\left(\tau^{(i)}\right)} \right] $$

Đến đây, ta nhận thấy có một vấn đề khác: Môi trường có tính ngẫu nhiên (các hàm phân phối trạng thái \\(P(s_{t+1} \mid s_t, a_t)\\)) và chính sách cũng mang tính ngẫu nhiên \\(\Rightarrow\\) cùng một trạng thái bắt đầu có thể dẫn đến các giá trị phần thưởng rất khác nhau \\(\Rightarrow\\) phần thưởng bắt đầu từ cùng một trạng thái có thể thay đổi đáng kể giữa các tập (episode) \\(\Rightarrow\\) phương sai giữa các tập cao. 

Giải pháp để giảm phương sai là sử dụng một số lượng lớn các quỹ đạo, tuy nhiên việc tăng kích thước (batch size) một cách đáng kể sẽ làm giảm hiệu quả sử dụng mẫu (sample efficiency).

> Hiệu quả sử dụng mẫu có thể hiểu là mức độ hiệu quả mà một mô hình học được từ một số lượng mẫu (dữ liệu) nhất định.

## Thuật toán REINFORCE

Kết hợp tất cả những thứ trên lại một cách đơn giản nhất, ta được thuật toán REINFORCE - là một dạng cụ thể của Policy Gradient. 

1. Khởi tạo bộ tham số \\(\theta\\) bất kỳ

2. Sử dụng chính sách \\(\pi_\theta\\) để thu thập một tập (episode) \\(\tau\\).

2. Sử dụng tập này để ước tính gradient \\(\nabla_\theta J(\theta)\\).

3. Cập nhật trọng số của chính sách: \\(\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)\\)

## Policy Gradient Theorem

Giờ ta quay lại chứng minh Policy Gradient Theorem nào!

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau}P(\tau;\theta)R(\tau) = \sum_{\tau} \nabla_\theta \left(P(\tau;\theta)R(\tau)\right) = \sum_{\tau} \nabla_\theta P(\tau;\theta) R(\tau) $$ 

$$ = \sum_{\tau} P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} R(\tau) = \sum_{\tau} P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) $$

$$ = \mathbb{E}_{\tau \sim \pi_\theta} \nabla_\theta \log \textcolor{blue}{P(\tau; \theta)} R(\tau) $$

Ta tiếp tục rút gọn \\(\nabla_\theta \log \textcolor{blue}{P(\tau; \theta)}\\). 

Với \\(\mu(s_0)\\) là phân phối trạng thái ban đầu và \\(P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right)\\) là MDP dynamics của một quỹ đạo \\(\tau^{(i)}\\) nào đó,

$$
\nabla_\theta \log \textcolor{blue}{P\left(\tau^{(i)}; \theta \right)} = \nabla_\theta \log \left[ \mu(s_0) \prod_{t=0}^{T} P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right) \pi_\theta \left(a_{t}^{(i)} \mid s_{t}^{(i)}\right) \right]
$$

$$
= \nabla_\theta \left[ \log \mu(s_0) + \sum_{t=0}^{T} \log P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right) + \sum_{t=0}^{T} \log \pi_\theta\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right) \right]
$$

$$ = \nabla_\theta \log \mu(s_0) + \nabla_\theta \sum_{t=0}^{T} \log P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right) + \nabla_\theta \sum_{t=0}^{T} \log \pi_\theta\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)
$$

$$
= 0 + 0 + \nabla_\theta \sum_{t=0}^{T} \log \pi_\theta\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)
= \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)
$$

Vậy 

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \left[ \textcolor{darkorange}{\nabla_\theta \log  \pi_\theta (a_t \mid s_t) R(\tau)}\right]\right] $$ 


## Nhận xét 

- Policy gradient đã giải quyết được vấn đề về không gian hành động liên tục mà Q-learning hay Deep Q-learning không làm được.
- Policy gradient có thể học một chính sách ngẫu nhiên. Ta không cần phải thực hiện đánh đổi khám phá/khai thác. Vì ta có một phân phối xác suất trên các hành động, nên tác nhân khám phá không gian trạng thái mà không phải lúc nào cũng đi theo cùng một quỹ đạo.
- Tuy nhiên, Policy gradient rất hay hội tụ đến tối ưu cục bộ thay vì tối ưu toàn cục.
- Có thể mất nhiều thời gian hơn để huấn luyện Policy gradient.
- Có thể có phương sai cao. 


