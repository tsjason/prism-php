<?php

declare(strict_types=1);

namespace Prism\Prism\Providers\Anthropic\Concerns;

use Illuminate\Http\Client\RequestException;
use Illuminate\Http\Client\Response;
use Prism\Prism\Contracts\PrismRequest;
use Prism\Prism\Exceptions\PrismException;

trait HandlesHttpRequests
{
    protected Response $httpResponse;

    /**
     * @return array<string, mixed>
     */
    abstract public static function buildHttpRequestPayload(PrismRequest $request): array;

    protected function sendRequest(): void
    {
        /** @var Response $response */
        $response = $this->client
            ->retry(
                times: 3,
                sleepMilliseconds: function (int $attempt, \Throwable $exception): int {
                    if ($exception instanceof RequestException && $exception->response->header('retry-after')) {
                        return (int) $exception->response->header('retry-after') * 1000;
                    }

                    return min(1000 * (2 ** ($attempt - 1)), 10000);
                },
                when: function (\Throwable $exception): bool {
                    if ($exception instanceof RequestException) {
                        return in_array($exception->response->getStatusCode(), [429, 500, 502, 503, 529]);
                    }

                    return false;
                },
                throw: true,
            )
            ->post(
                'messages',
                static::buildHttpRequestPayload($this->request)
            );

        $this->httpResponse = $response;

        $this->handleResponseErrors();
    }

    protected function handleResponseErrors(): void
    {
        $data = $this->httpResponse->json();

        if (data_get($data, 'type') === 'error') {
            throw PrismException::providerResponseError(vsprintf(
                'Anthropic Error: [%s] %s',
                [
                    data_get($data, 'error.type', 'unknown'),
                    data_get($data, 'error.message'),
                ]
            ));
        }
    }
}
