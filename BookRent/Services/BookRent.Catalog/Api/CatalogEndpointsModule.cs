using BookRent.Catalog.Infrastructure.Interfaces;

namespace BookRent.Catalog.Api;

internal static class CatalogEndpointsModule
{
    internal static void MapCatalogEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/catalog");

        group.MapGet("/books", CatalogCustomerEndpoints.GetAllBooks);
        group.MapGet("/books/search", CatalogCustomerEndpoints.SearchBook);
        group.MapGet("/book/{id:guid}", CatalogCustomerEndpoints.GetBook);
        group.MapPost("/addBooks", CatalogEmployeeEndpoints.AddBooks);
        group.MapPost("/removeBooks", CatalogEmployeeEndpoints.RemoveBooks);
    }
}