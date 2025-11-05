
SELECT IFNULL(
    (SELECT DISTINCT salary
    FROM Employee
    ORDER BY Salary DESC
    LIMIT 1 OFFSET 1), # 显示一个值，显示之前跳过一个
    NULL
) AS SecondHighestSalary