# Write your MySQL query statement below
with new_table as (
    select
    d.name as "Department",
    e.name as "Employee",
    e.salary as "Salary",
    dense_rank() over(partition by d.name
                        order by e.salary desc) as ranking
    from Employee e
    left join Department d
    on e.departmentId = d.id
)
select Department, Employee, Salary
from new_table
where ranking <= 3


-- DENSE_RANK() OVER 是一个 SQL 窗口函数，用于在结果集中为每一行分配一个连续的排名，该排名基于 OVER 子句中指定的 PARTITION BY 和 ORDER BY 子句。与 RANK() 相比，DENSE_RANK() 的排名是连续的，而 RANK() 的排名会有跳跃。 
-- 用法
-- 基本语法是：
-- DENSE_RANK() OVER (PARTITION BY column1 ORDER BY column2)
-- DENSE_RANK(): 窗口函数本身。
-- OVER (...): 指定窗口函数的操作范围。
-- PARTITION BY column1: 可选。将数据划分为不同的分区，DENSE_RANK() 会在每个分区内独立进行排名。
-- ORDER BY column2: 必需。指定用于排序的列，排名将根据此列的值进行。 
-- 示例
-- 假设有一个 Students 表，包含 Name 和 Score 列。我们想为学生按照分数进行连续排名。
-- sql
-- SELECT
--     Name,
--     Score,
--     DENSE_RANK() OVER (ORDER BY Score DESC) AS Rank
-- FROM
--     Students;
-- DENSE_RANK() 与 RANK() 的区别
-- 特性 	DENSE_RANK()	RANK()
-- 排名	连续，没有跳跃	可能会跳跃
-- 示例	如果有并列的最高分，下一个分数将是排名第二	如果有并列的最高分，下一个分数将是排名第四
