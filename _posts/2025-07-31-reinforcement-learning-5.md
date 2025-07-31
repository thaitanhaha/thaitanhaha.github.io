---
title: 'Reinforcement learning 5 - Trust Region Policy Optimization'
date: 2025-07-31
permalink: /posts/reinforcement-learning-5/
tags:
  - reinforcement learning
---

Các phương pháp Policy Gradient có thể dẫn đến các cập nhật chính sách quá lớn, khiến cho mô hình bị "lạc" khỏi những gì đã học được trong quá khứ và trở nên không ổn định. Điều này có thể làm cho quá trình huấn luyện trở nên kém hiệu quả.

Giống như trong gradient descent hay gradient ascent, nếu bước đi quá nhỏ, mô hình sẽ học rất chậm; còn nếu bước đi quá lớn, nó có thể đi sai hướng và mất ổn định.

<!-- Mỗi một lần cập nhật chính sách, ta phải lấy mẫu toàn bộ một quỹ đạo, nói cách khác là cập nhật các trạng thái trong quỹ đạo đó. Các trạng thái trong quỹ đạo thì ít nhiều có sự tương đồng nhau, nên những thay đổi này sẽ chồng lắp lẫn nhau, làm cho quá trình huấn luyện trở nên rất nhạy cảm và không ổn định. -->

## Natural policy gradients

Nói chung, ta muốn chính sách được cập nhật sao cho $$\underbrace{\text{hàm mục tiêu thì vừa tăng}}_{1}$$, vừa đảm bảo $$\underbrace{\text{chính sách mới và cũ không quá khác biệt}}_{2}$$.

### 1 - \\(\mathcal{L}_{\theta}(\theta')\\)

Gọi \\(\mathcal{L}_{\theta}(\theta')\\) là hàm mục tiêu ước tính sự cải thiện trong phần thưởng kỳ vọng khi chuyển từ \\(\pi_\theta\\) sang \\(\pi_{\theta'}\\). Với \\(A(s_t,a_t)\\) là hàm lợi thế thì

$$ \mathcal{L}_{\theta}(\theta') = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t \dfrac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\theta}(a_t \mid s_t)} A^{\pi_{\theta}}(s_t, a_t)\right]$$

    
### 2 - \\(\mathcal{\bar{D}}_{KL}\\)

Vấn đề ở bước đi quá nhỏ hay quá lớn là vì ta lấy đạo hàm bậc nhất, nó chỉ cho chúng ta biết nên bước theo hướng nào, nhưng không cho biết bước đi đó phải lớn bao nhiêu \\(\Rightarrow\\) Nếu đang ở một đường cong, đạo hàm bậc nhất đưa ta đi rất xa \\(\Rightarrow\\) Để khắc phục vấn đề đó, Natural policy gradients tính luôn đạo hàm bậc hai. 

Để làm điều này, chúng ta tính toán sự khác biệt giữa chính sách trước và sau khi cập nhật, mà chính sách là phân phối xác suất, nên sự khác biệt này chính là KL-divergence. Như đã nói, ta không muốn bước cập nhật khiến cho chính sách trở nên quá khác biệt, nên ta giới hạn KL-divergence lại bằng \\(\delta\\).

$$ \theta_{k+1} = \underset{\theta_k + \Delta \theta}{\operatorname{argmax}} \mathcal{L}_{\theta_k}(\theta_k + \Delta \theta)$$ 

$$ \text{s.t. } \mathcal{\bar{D}}_{KL}(\theta_k + \Delta \theta \mid \mid \theta_k) \leq \delta $$

$$\text{với } \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k) = \mathbb{E}_{s \sim \pi_{\theta_k}} \mathcal{D}_{KL}(\pi_{\theta}(. | s) \mid \mid \pi_{\theta_k}(. | s)) $$

Để giải quyết phương trình này, ta có thể sử dụng chuỗi Taylor để mở rộng cả hai hạng tử trên đến bậc hai. 

> Dưới đây ta quy ước \\(f(x) \mid _{x_0}\\) là \\(f(x_0)\\)

$$ \mathcal{L}_{\theta_k}(\theta) = \mathcal{L}_{\theta_k}(\theta_k) + (\nabla_\theta \mathcal{L}_{\theta_k}(\theta) \mid_{\theta_k})^\top \cdot (\theta - \theta_k) $$ 

$$ = (\nabla_\theta \mathcal{L}_{\theta_k}(\theta) \mid_{\theta_k})^\top \cdot (\theta - \theta_k) $$

$$ \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k) = \mathcal{\bar{D}}_{KL}(\theta_k \mid \mid \theta_k) + \nabla_\theta \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k)\mid_{\theta_k} \cdot (\theta - \theta_k) + \dfrac{1}{2}(\theta - \theta_k)^\top \cdot \nabla^2_\theta \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k)\mid_{\theta_k} \cdot (\theta - \theta_k) $$

$$ = \dfrac{1}{2}(\theta - \theta_k)^\top \cdot \nabla^2_\theta \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k)\mid_{\theta_k} \cdot (\theta - \theta_k) $$

Bài toán trên trở thành

$$ \theta_{k+1} = \underset{\theta}{\operatorname{argmax} \,} (\underbrace{\nabla_\theta \mathcal{L}_{\theta_k}(\theta)  \mid_{\theta_k}}_{\textcolor{blue}{g}})^\top \cdot (\theta - \theta_k) $$ 

$$ \text{s.t. } \quad \dfrac{1}{2}(\theta - \theta_k)^\top \cdot \underbrace{\nabla^2_\theta \mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k) \mid_{\theta_k}}_{\textcolor{purple}{H}} \cdot (\theta - \theta_k) \leq \delta $$

Giải phương trình trên, ta thu được

$$ \theta_{k+1} = \theta_k + \textcolor{orange}{\sqrt{\dfrac{2 \delta}{g^\top H^{-1} g}} H^{-1}g} $$

Về mặt tính toán, tính \\(H\\) đã khó, tính \\(H^{-1}\\) còn khó hơn, \\(\mathcal{O}(n^3)\\), nên người ta dùng Truncated Natural Policy Gradient để ước tính \\(x = H^{-1}g\\). Cụ thể hơn, ta giải phương trình \\(Hx = g\\) với thuật toán Conjugate Gradient. 

<details><summary markdown="span">Thuật toán Conjugate Gradient</summary>

[**Thuật toán Conjugate Gradient**](https://www.math.hkust.edu.hk/~mamu/courses/531/cg.pdf) để giải \\(Ax=b\\) 

Gọi \\(x_0 = 0\\), \\(r_0 = Ax_0 - b\\) và \\(p_0 = -r_0\\).

Với \\(k\\) chạy từ 0, 

1. \\(\alpha_k = \dfrac{r_k^\top r_k}{p_k^\top A p_k}\\)
2. \\(x_{k+1} = x_k + \alpha_k  p_k\\)
3. \\(r_{k+1} = r_k - \alpha_k  A  p_k\\)
4. \\(\beta_k = \dfrac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}\\)
5. \\(p_{k+1} = -r_{k+1} + \beta_k  p_k\\)

Kết quả: \\(x_{k+1}\\) là nghiệm gần đúng của \\(Ax = b\\)
</details>

Khi đó, 

$$\textcolor{orange}{\sqrt{\dfrac{2 \delta}{g^\top H^{-1} g}} H^{-1}g} = \sqrt{\dfrac{2\delta}{x^T H x}} x$$

### Tổng kết lại

Natural policy gradients chạy vòng lặp với \\(k=0,1,\dots\\):

1. Thu thập tập các quỹ đạo $$\mathcal{D}_k$$ theo chính sách $$\pi_{\theta_k}$$
2. Ước lượng hàm lợi thế $$A_t^{\pi_{\theta_k}}$$
3. Tính các ước lượng mẫu cho \\(g_k\\) 
4. Sử dụng Conjugate Gradient để tìm $$x_k \approx H_k^{-1} g_k$$
5. Cập nhật tham số chính sách $$\theta_{k+1} = \theta_k + \sqrt{\dfrac{2\delta}{x_k^T H_k x_k}} x_k$$

## Trust Region Policy Optimization (TRPO)

Vấn đề là, do các sai số xấp xỉ được tạo ra bởi khai triển Taylor, điều này có thể không thỏa mãn được ràng buộc KL. TRPO thêm một sửa đổi vào quy tắc cập nhật này: **line search**. 

Tức là, giảm dần kích thước của cập nhật theo từng bước cho đến khi tìm được cập nhật đầu tiên không vi phạm ràng buộc. Quy trình này có thể được xem như là thu nhỏ vùng tin cậy, tức là vùng mà trong đó chúng ta tin rằng cập nhật thực sự sẽ cải thiện mục tiêu.

Với thuật toán trên, ở bước tính \\(\Delta_k = \sqrt{\dfrac{2\delta}{x_k^T H_k x_k}} x_k\\). Để giảm dần vùng tin cậy, ta dùng một $0 < \alpha < 1$. Với \\(j = 0,1,\dots,L\\)

1. Tính \\(\theta = \theta_k + \alpha^j \Delta_k\\)
2. Nếu $$\mathcal{L}_{\theta_k}(\theta) \ge 0$$ và $$\mathcal{\bar{D}}_{KL}(\theta \mid \mid \theta_k) \le \delta$$ thì dừng vòng lặp, chọn \\(\theta_{k+1} = \theta_k + \alpha^j \Delta_k\\). 

## Nhận xét

- TRPO rất nặng về toán và lý thuyết. Bài viết này cũng đã lượt bỏ khá nhiều phần chứng minh đằng sau các công thức.

- Tuy nhiên, chính điều đó đóng góp nền tảng lớn cho các phương pháp sau, tiêu biểu là Proximal Policy Optimization - PPO.

- Việc triển khai TRPO yêu cầu tính toán phức tạp và tốn kém về thời gian và tài nguyên. Các thuật toán yêu cầu việc tính toán Hessian và giải hệ phương trình tuyến tính với độ phức tạp tính toán cao, điều này khiến TRPO ít được áp dụng trong các bài toán quy mô lớn hoặc trong môi trường không có tài nguyên tính toán mạnh.








