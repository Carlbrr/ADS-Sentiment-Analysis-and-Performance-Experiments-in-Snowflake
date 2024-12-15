select current_version(); --8.41.1
-- obs, we run these from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 (and then scaling factor 1, 10, 100, 1000)
-- as well as  four different sizes of virtual warehouses (XS, S, M, and L). Measure the query runtimes.

--Make sure to set AUTO_SUSPEND = 10 when creating a new warehouse. This means the warehouse will be suspended after 10 seconds of inactivity. The default value is 600 seconds. Setting it to a smaller value helps save money because we pay for running warehouses.

--https://docs.snowflake.com/en/sql-reference/sql/create-warehouse
create or replace warehouse bullfrog_wh WAREHOUSE_SIZE = LARGE, AUTO_SUSPEND = 40;
--alter warehouse bullfrog_wh set WAREHOUSE_SIZE = SMALL;
--use warehouse bullfrog_wh; -- "Creating a virtual warehouse automatically sets it as the warehouse in use for the current session"

ALTER SESSION SET USE_CACHED_RESULT=FALSE;
-- VIEW CURRENT PARAMETER SETTING
SHOW PARAMETERS LIKE '%CACHE%';

-- https://examples.citusdata.com/tpch_queries.html
-- og query 18 fra http://www.qdpma.com/tpch/TPCH100_Query_plans.html
-- QUERY 1 (scan-intensive)
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
FROM
    lineitem
WHERE
    l_shipdate <= DATEADD(day, -90, DATE '1998-12-01') -- replaced: date '1998-12-01' - interval '90' day (snowflake does the intervals a bit differently)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;


-- QUERY 5 (join-intensive)
SELECT
    n_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue
FROM
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'ASIA'
    AND o_orderdate >= date '1994-01-01'
    AND o_orderdate < DATEADD(year, 1, DATE '1994-01-01') -- replaced: date '1994-01-01' + interval '1' year
GROUP BY
    n_name
ORDER BY
    revenue desc;

-- QUERY 18 (aggregate-intensive query (involving many group-by operations))
SELECT TOP 100 C_NAME, C_CUSTKEY, O_ORDERKEY, O_ORDERDATE, O_TOTALPRICE, SUM(L_QUANTITY)
FROM CUSTOMER, ORDERS, LINEITEM
WHERE O_ORDERKEY IN (SELECT L_ORDERKEY FROM LINEITEM GROUP BY L_ORDERKEY HAVING
 SUM(L_QUANTITY) > 300) AND C_CUSTKEY = O_CUSTKEY AND O_ORDERKEY = L_ORDERKEY
GROUP BY C_NAME, C_CUSTKEY, O_ORDERKEY, O_ORDERDATE, O_TOTALPRICE
ORDER BY O_TOTALPRICE DESC, O_ORDERDATE;
