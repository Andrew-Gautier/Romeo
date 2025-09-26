SELECT 
    vuln,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM funcs 
WHERE (end - start) BETWEEN 10 AND 50
GROUP BY vuln
ORDER BY vuln;

SELECT COUNT(*) 
FROM funcs 
WHERE LENGTH(code) - LENGTH(REPLACE(code, '\n', '')) + 1 BETWEEN 10 AND 150;


SELECT 
    vuln,
    COUNT(*) as func_count
FROM funcs 
WHERE (LENGTH(code) - LENGTH(REPLACE(code, '\n', '')) + 1) BETWEEN 10 AND 150
  AND vuln IN ('0', '1')
GROUP BY vuln
ORDER BY vuln;