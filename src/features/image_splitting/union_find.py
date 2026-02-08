"""
Union-Find (並查集) 資料結構

用於高效處理連通分量問題
"""


class UnionFind:
    """
    靜態 Union-Find 資料結構

    用於固定大小的元素集合，支援路徑壓縮和按秩合併優化
    """

    def __init__(self, size: int) -> None:
        """
        初始化 Union-Find

        Args:
            size: 元素數量 (0..size-1)
        """
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        """
        尋找元素的根節點 (帶路徑壓縮)

        Args:
            x: 元素索引

        Returns:
            根節點索引
        """
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路徑壓縮
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        合併兩個元素所在的集合

        Args:
            a: 第一個元素索引
            b: 第二個元素索引
        """
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return

        rank = self._rank
        parent = self._parent

        # 按秩合併：將較小的樹連接到較大的樹
        if rank[pa] < rank[pb]:
            parent[pa] = pb
            return
        if rank[pa] > rank[pb]:
            parent[pb] = pa
            return

        parent[pb] = pa
        rank[pa] += 1


class UnionFindDynamic:
    """
    動態 Union-Find 資料結構

    支援動態增加新元素，使用 1-based ID (ID 0 保留)
    """

    def __init__(self) -> None:
        """初始化動態 Union-Find"""
        self._parent: list[int] = [0]
        self._rank: list[int] = [0]

    def make_set(self) -> int:
        """
        建立新集合

        Returns:
            新集合的 ID
        """
        new_id = len(self._parent)
        self._parent.append(new_id)
        self._rank.append(0)
        return new_id

    def find(self, x: int) -> int:
        """
        尋找元素的根節點 (帶路徑壓縮)

        Args:
            x: 元素 ID

        Returns:
            根節點 ID
        """
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路徑壓縮
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        合併兩個元素所在的集合

        Args:
            a: 第一個元素 ID
            b: 第二個元素 ID
        """
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return

        rank = self._rank
        parent = self._parent

        # 按秩合併
        if rank[pa] < rank[pb]:
            parent[pa] = pb
            return
        if rank[pa] > rank[pb]:
            parent[pb] = pa
            return

        parent[pb] = pa
        rank[pa] += 1
