token 前缀 → slot_id → KV 池中的 K/V 映射链路

## 先分清四种东西

| 名称 | 含义 |
|---|---|
| `request_row` | 当前活跃请求在 `page_table` 中使用的行号 |
| `logical_position` | token 在这个请求序列中的位置，例如第 0、1、2 个 token |
| `slot_id` | KV 池内一个 token 槽位的整数编号 |
| CUDA 虚拟地址 | CUDA tensor 对应的 GPU 地址，kernel 用它访问 tensor；`slot_id` **不是**这个地址 |

Mini-SGLang 的 KV 池形状是：

```text
[ K_or_V, layer, page_id, offset_in_page, kv_head, head_dim ]
```

在代码里，K 和 V 是分开的 view，底层由同一个 GPU tensor 分配出来。[KV 池分配]( /home/po/ai/mini-sglang/python/minisgl/kvcache/mha_pool.py:28)

`slot_id` 用来选定某个 **page 中的 token 位置**：

```text
page_id       = slot_id // page_size
offset_in_page = slot_id % page_size
```

因此：

```python
page_table[request_row, logical_position] == slot_id
```

表示：**这个请求在 `logical_position` 处的 token，其 K/V 写在 KV 池由 `slot_id` 指定的位置。**

## `slot_id` 怎么定位到 K/V

假设 `page_size = 4`，某个请求的映射行是：

```text
logical_position:  0   1   2   3 | 4   5   6   7
page_table[row]:  12  13  14  15 | 32  33  34  35
```

对 logical position `5`：

```text
slot_id = page_table[row, 5] = 33
page_id = 33 // 4 = 8
offset  = 33 % 4  = 1
```

假设此时正在访问第 2 层、第 3 个 KV head，那么对应的 key 和 value 是：

```python
key   = K_pool[2, 8, 1, 3, :]
value = V_pool[2, 8, 1, 3, :]
```

也可以把某层的 K 池展平成 `[所有 token slots, kv_head, head_dim]` 来理解：

```python
key = K_pool[layer].view(-1, kv_heads, head_dim)[slot_id, kv_head, :]
```

`slot_id` 只是 tensor 中的索引。tensor 的 CUDA 虚拟地址加上相应的 tensor 偏移，才让 kernel 访问到该元素；GPU 硬件再负责虚拟地址到显存物理位置的转换。

## `page_table` 和 KV 池各自负责什么

```text
page_table[row, logical_position] -> slot_id
slot_id                         -> KV 池中的 page 和 offset
(page, offset, layer, head)     -> K 或 V tensor 元素
```

所以 `page_table`：

- **存映射索引**，不存 K/V 数值；
- **不存 CUDA 地址**；
- 每行对应一个当前活跃请求，列对应该请求的逻辑 token 位置。

KV 池则保存实际 K/V 数值。Mini-SGLang 初始化时分配 GPU 上的 `page_table`；它的物理 KV allocator 为新 token 分配 slot，再把 slot 写入对应请求行。[`page_table` 分配]( /home/po/ai/mini-sglang/python/minisgl/engine/engine.py:65) [分配 slot 并更新映射]( /home/po/ai/mini-sglang/python/minisgl/scheduler/cache.py:42)

## Radix cache 在哪里接进来？

Radix cache 用 token 前缀找到已有的 slot ID：

```text
Radix tree:
    token prefix -> slot IDs

GPU page_table:
    request_row + logical position -> slot ID

GPU KV pool:
    slot ID + layer + head -> K/V 数值
```

新请求命中前缀后，scheduler 把匹配到的 slot IDs 放入这次请求新分配的 row；缺失后缀则分配新 slot。接着 attention kernel 用整行 page table 读取历史 KV、写入新 KV。KV 写入使用的 `out_loc` 正是这些位置索引。[命中索引写到新 row]( /home/po/ai/mini-sglang/python/minisgl/scheduler/prefill.py:56) [attention 根据索引写入 KV]( /home/po/ai/mini-sglang/python/minisgl/attention/fi.py:176)

一句话总结：**Radix cache 负责从 token 前缀找到 slot IDs；page table 负责按请求和逻辑位置组织 slot IDs；KV pool 负责按 slot ID 存取每层 K/V 数值。**