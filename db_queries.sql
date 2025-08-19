SELECT 
    vuln,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM funcs 
WHERE (end - start) BETWEEN 10 AND 50
GROUP BY vuln
ORDER BY vuln;