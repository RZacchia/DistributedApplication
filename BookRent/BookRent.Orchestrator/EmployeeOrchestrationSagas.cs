using BookRent.Orchestrator.Api.Requests;
using BookRent.Orchestrator.Clients;
using BookRent.Orchestrator.Services.Interfaces;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;

namespace BookRent.Orchestrator;

internal static class EmployeeOrchestrationSagas
{

    public static async Task<IResult> RemoveBookFromCatalogSaga([FromBody] RemoveFromCatalogRequest request, ICatalogService service)
    {
        return await service.RemoveFromCatalogSaga(request);
    }
    
    
}