import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';
import exec from 'k6/execution';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:9999';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('./test-data.json')).entries;
});
const statsArr = new SharedArray('test-stats', function () {
    return [JSON.parse(open('./test-data.json')).stats];
});
const expectedStats = statsArr[0];

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['p(95)', 'p(99)'],
    systemTags: ['status', 'method'],
    dns: {
        ttl: '5m',
        select: 'roundRobin',
    },
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 80,
            maxVUs: 180,
            gracefulStop: '5s',
            stages: [
                { duration: '15s', target: 250 },
                { duration: '15s', target: 400 },
                { duration: '10s', target: 500 },
            ],
        },
    },
};

export function setup() {
    console.log(
        `Stress dataset: ${expectedStats.total} entries, `
        + `${expectedStats.fraud_count} fraud, `
        + `${expectedStats.legit_count} legit`
    );
}

export default function () {
    const idx = exec.scenario.iterationInTest;
    if (idx >= testData.length) return;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        `${BASE_URL}/fraud-score`,
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1);
            else tpCount.add(1);
        } else {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

export function handleSummary(data) {
    const httpDuration = data.metrics.http_req_duration.values;
    const tp = data.metrics.tp_count ? data.metrics.tp_count.values.count : 0;
    const tn = data.metrics.tn_count ? data.metrics.tn_count.values.count : 0;
    const fp = data.metrics.fp_count ? data.metrics.fp_count.values.count : 0;
    const fn = data.metrics.fn_count ? data.metrics.fn_count.values.count : 0;
    const errs = data.metrics.error_count ? data.metrics.error_count.values.count : 0;
    const total = tp + tn + fp + fn + errs;

    const result = {
        profile: 'stress',
        executed: total,
        expected: expectedStats,
        p95: httpDuration['p(95)'].toFixed(2) + 'ms',
        p99: httpDuration['p(99)'].toFixed(2) + 'ms',
        breakdown: {
            true_positive_detections: tp,
            true_negative_detections: tn,
            false_positive_detections: fp,
            false_negative_detections: fn,
            http_errors: errs,
        },
    };

    return {
        'test/stress-results.json': JSON.stringify(result, null, 2),
    };
}
