using BookRent.Orchestrator.Api.Requests;

namespace BookRent.Orchestrator.Services.Interfaces;

public interface ICatalogService
{
    Task<IResult> RemoveFromCatalogSaga(RemoveFromCatalogRequest request);
}