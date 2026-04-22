# RWKV_Evolve
---
#####  SIG:  <函数签名，含输入输出类型>
#####  DEF:  <形式化规则，用数学符号而非自然语言>
##### LIM:  <规模 + 数值边界>
##### TRAP: <非显然陷阱，无则省略>


---
##### SIG:  isValidBST(root: TreeNode) -> bool
##### DEF:  ∀ v ∈ tree:  max(v.left) < v.val < min(v.right)；子树递归满足
##### LIM:  n ∈ [1, 10⁴]；val ∈ [-2³¹, 2³¹-1]
##### TRAP: 严格 < 非 ≤；val 可触 INT 边界（哨兵法溢出）；父子局部比较不足，须子树全局约束

---

##### SIG:  twoSum(nums: int[], target: int) -> int[2]
##### DEF:  ∃! (i, j), i ≠ j:  nums[i] + nums[j] = target，返回 [i, j]
##### LIM:  n ∈ [2, 10⁴]；nums[i], target ∈ [-10⁹, 10⁹]
##### TRAP: 同一元素不可用两次；解保证唯一存在（无需处理无解 / 多解）

---
##### 维度 A — 计算范式（how you compute）：暴力枚举 / 分治 / 贪心 / 动态规划 / 回溯-剪枝 / 网络流-线性规划 / 随机化 / 归约到已知问题

---
##### 维度 B — 数据结构（what you remember）：数组 / 哈希 / 栈队列 / 堆 / 链表 / 树（BST、线段树、树状数组、并查集、Trie）/ 图邻接表 / 位图 / 跳表 / 持久化结构

---
##### 维度 C — 搜索/遍历模式（how you traverse）：顺序扫描 / 双指针 / 滑动窗口 / 二分 / BFS / DFS / 拓扑序 / 扫描线 / Meet-in-the-middle

---
##### 一道具体题的解 = A × B × C 的一个点（或少数几个点的组合）。例如"滑动窗口最大值" = (扫描型计算, 单调队列, 滑动窗口) = (A=线性扫描, B=单调双端队列, C=窗口)。

---

## case 1: Error and fix
## case 2: Passing and optimize/evolve

## One Shot得到
## Â = {扫描式贪心} B̂ = {堆, 单调队列, 有序集} Ĉ = {滑窗}

## 然后从简单的开始Evolve

## Each thinking format:
### 每一个think包括instruction -> instructionthinking的格式
### 1. Question Information
### 2. One Shot strategies
### 3.Best implementation obtained
### 4. Best Implementation Strategy
### 5. Current Strategy Implementation

## Question -> Question information converstion

## Question information converstion -> One Shot strategies

## product: Algorithm Searcher with Typescript and Rust encoding